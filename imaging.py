from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import nibabel as nib
import numpy as np


# ``labelcomparison.py`` in the existing QC workflow deliberately presents
# the image in L-P-I display orientation.  The workbook coordinates are still
# physical RAS coordinates; this constant only describes the voxel order used
# by the viewer after reorientation.  Keeping the display affine alongside
# the display array is what makes RAS -> voxel navigation reversible.
DISPLAY_AXCODES = ("L", "P", "I")

# Display presets differ only in the left-right axis, so the axis *roles* are
# identical in both: axis 0 is left-right, axis 1 is anterior-posterior and
# axis 2 is superior-inferior.  Slice axes, plane extraction and wheel
# navigation are therefore preset-independent, and the physical meaning of a
# voxel is carried entirely by the reoriented affine.
ORIENTATION_PRESETS: dict[str, dict[str, object]] = {
    "radiological": {
        "label": "Radiological",
        "axcodes": ("L", "P", "I"),
        "summary": "Patient right on the left of the image",
    },
    "neurological": {
        "label": "Neurological",
        "axcodes": ("R", "P", "I"),
        "summary": "Patient left on the left of the image",
    },
}
DEFAULT_ORIENTATION = "radiological"

_OPPOSITE_AXCODE = {"L": "R", "R": "L", "A": "P", "P": "A", "S": "I", "I": "S"}


def preset_axcodes(preset: str | None) -> tuple[str, str, str]:
    """Display axis codes for a named orientation preset."""

    entry = ORIENTATION_PRESETS.get(str(preset or "").lower())
    if entry is None:
        entry = ORIENTATION_PRESETS[DEFAULT_ORIENTATION]
    return tuple(entry["axcodes"])  # type: ignore[return-value]


def opposite_axcode(code: str) -> str:
    try:
        return _OPPOSITE_AXCODE[str(code).upper()]
    except KeyError as exc:
        raise ValueError(f"Unknown anatomical axis code: {code}") from exc


def plane_direction_labels(
    axcodes: Iterable[str],
    column_axis: int,
    row_axis: int,
) -> tuple[str, str, str, str]:
    """Return (left, right, top, bottom) labels for a displayed plane.

    The labels are derived from the axis codes of the array that is actually on
    screen, so a direction label cannot disagree with the pixels next to it.
    An axis code names the direction of *increasing* index, which is drawn
    towards the right and towards the bottom.
    """

    codes = [str(code).upper() for code in axcodes]
    column = codes[int(column_axis)]
    row = codes[int(row_axis)]
    return opposite_axcode(column), column, opposite_axcode(row), row


@dataclass(frozen=True)
class Volume:
    """A volume loaded in the viewer's canonical L-P-I display orientation.

    ``affine`` maps these display voxels back into the original physical RAS
    world.  It is therefore the only affine that should be used for both
    microbleed coordinates and manual RAS jumps.
    """

    path: str
    data: np.ndarray
    affine: np.ndarray
    shape: tuple[int, ...]
    voxel_sizes: tuple[float, ...]
    orientation: tuple[str, str, str] = DISPLAY_AXCODES
    # Grayscale window for the whole volume.  Computing it once at load time
    # keeps the contrast identical in all three views and avoids repeating a
    # full-volume percentile for every plane and every modality switch.
    window: tuple[float, float] | None = None
    # Geometry of the file as it sits on disk, before reorientation.  Anything
    # written back out -- a segmentation, for instance -- has to use this, so
    # the result is voxel-for-voxel identical to the image it belongs to.
    source_affine: np.ndarray | None = None
    source_shape: tuple[int, ...] | None = None
    # ``(sform_code, qform_code)`` of the file, so a label written back out
    # declares the same space as the image it belongs to.
    source_codes: tuple[int, int] | None = None


# Windows marks a file whose contents are not on this disk yet.  A study on
# OneDrive is mostly such files -- 124 of the 500 sequence files here on the
# machine this was written on -- and the first read of one blocks until the
# whole thing has been downloaded.
FILE_ATTRIBUTE_OFFLINE = 0x00001000
FILE_ATTRIBUTE_RECALL_ON_OPEN = 0x00040000
FILE_ATTRIBUTE_RECALL_ON_DATA_ACCESS = 0x00400000
CLOUD_ATTRIBUTES = (
    FILE_ATTRIBUTE_OFFLINE
    | FILE_ATTRIBUTE_RECALL_ON_OPEN
    | FILE_ATTRIBUTE_RECALL_ON_DATA_ACCESS
)


def is_cloud_only(path: str | Path) -> bool:
    """True when reading this file means downloading it first.

    Cheap: one stat, no open.  Everything that wants to warn a reader before
    a several-second stall has to ask this about every file first, including
    the ones that turn out to be local.
    """

    try:
        attributes = getattr(os.stat(path), "st_file_attributes", 0)
    except OSError:
        return False
    return bool(attributes & CLOUD_ATTRIBUTES)


def fetch_local_copy(path: str | Path, chunk_bytes: int = 4 * 1024 * 1024) -> int:
    """Pull a cloud file onto the disk, and report how many bytes that was.

    Reading and discarding is the only way to ask Windows for the contents;
    there is no "download this" call that does not go through a read.  Done
    before the loader so the several seconds of waiting happen somewhere a
    progress dialog can be on screen -- measured at 2.05 s for a 28.6 MB
    sequence here, and the download is *not* incremental: the first 4 MB read
    took 2.04 of those seconds and the rest returned instantly.  So this
    reports whole files, never a percentage within one, because a percentage
    within one would be a lie.
    """

    total = 0
    with open(path, "rb") as handle:
        while True:
            block = handle.read(chunk_bytes)
            if not block:
                break
            total += len(block)
    return total


def load_volume(path: str | Path, axcodes: Iterable[str] | None = None) -> Volume:
    """Load a NIfTI volume in the viewer's display space.

    The workbook stores physical RAS coordinates.  Reorienting the image and
    retaining the reoriented affine preserves that physical coordinate space;
    the display voxel indices simply change to the requested array order.  This
    is important because using an RAS-oriented array for one operation and a
    display-oriented array for another makes a manual jump land on a different
    anatomical location.

    ``axcodes`` selects the display orientation preset.  Whichever preset is
    used, the returned ``affine`` is the one that maps those display voxels
    back to physical RAS, so it is the only thing callers need in order to stay
    correct.
    """

    display_axcodes = tuple(str(code).upper() for code in (axcodes or DISPLAY_AXCODES))
    if len(display_axcodes) != 3:
        raise ValueError("Display orientation needs exactly three axis codes.")
    file_path = Path(path)
    image = nib.load(str(file_path))
    current = nib.orientations.io_orientation(image.affine)
    target = nib.orientations.axcodes2ornt(display_axcodes)
    transform = nib.orientations.ornt_transform(current, target)
    display_image = image.as_reoriented(transform)
    # ``get_fdata`` applies NIfTI scaling and guarantees that the array is
    # actually materialized as numeric data.  This avoids lazy proxy edge
    # cases where an apparently valid slice can be rendered as black after a
    # state transition.
    data = np.asarray(display_image.get_fdata(dtype=np.float32))
    if data.ndim > 3:
        data = data[..., 0]
    if data.ndim != 3:
        raise ValueError(f"Expected a 3D NIfTI volume, found shape {data.shape}.")
    affine = np.asarray(display_image.affine, dtype=np.float64)
    zooms = tuple(float(x) for x in display_image.header.get_zooms()[:3])
    return Volume(
        path=str(file_path),
        data=data,
        affine=affine,
        shape=tuple(int(x) for x in data.shape),
        voxel_sizes=zooms,
        orientation=display_axcodes,
        window=robust_window(subsample(data)),
        source_affine=np.asarray(image.affine, dtype=np.float64),
        source_shape=tuple(int(x) for x in image.shape[:3]),
        source_codes=_space_codes(image),
    )


def _space_codes(image) -> tuple[int, int] | None:
    """``(sform_code, qform_code)`` of a NIfTI, when it has them."""

    header = getattr(image, "header", None)
    try:
        return int(header["sform_code"]), int(header["qform_code"])
    except (TypeError, KeyError, ValueError):
        return None


def to_source_orientation(labels: np.ndarray, reference: Volume):
    """Turn a display-oriented array back into the source file's own geometry.

    The viewer works in a display orientation (L-P-I by default), so a mask
    painted here is in that voxel order.  Written out as-is it would carry a
    flipped affine relative to the image it belongs to: correct in world space,
    but not voxel-for-voxel aligned with the original NIfTI, which is what
    every other tool expects when it loads an image and its label together.
    Reorienting back removes that whole class of confusion.
    """

    array = np.asarray(labels)
    if array.shape != tuple(reference.shape):
        raise ValueError(
            f"Label shape {array.shape} does not match the image {tuple(reference.shape)}."
        )
    display = nib.Nifti1Image(array, np.asarray(reference.affine, dtype=np.float64))
    if reference.source_affine is None:
        return display
    current = nib.orientations.axcodes2ornt(tuple(reference.orientation))
    target = nib.orientations.io_orientation(np.asarray(reference.source_affine))
    return display.as_reoriented(nib.orientations.ornt_transform(current, target))


def save_label_volume(path: str | Path, labels: np.ndarray, reference: Volume) -> Path:
    """Write a label volume in the geometry of the image it was drawn on.

    Both ``sform`` and ``qform`` are set: some tools read one, some the other,
    and a label whose qform is left unset is a classic source of "the mask is
    in the wrong place" in a different viewer.

    Written to a neighbour first and renamed over the target, because this one
    file holds every mask this reader has drawn for the case -- twenty-five of
    them on the busiest case here.  Writing in place means a crash, a full
    disk, or the sync client holding the handle at the wrong moment leaves a
    truncated file where all of that work was.  ``os.replace`` within a
    directory is atomic, so the old file survives intact until the new one is
    complete.
    """

    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    restored = to_source_orientation(labels, reference)
    affine = np.asarray(restored.affine, dtype=np.float64)
    image = nib.Nifti1Image(np.asarray(restored.dataobj).astype(np.uint16), affine)
    image.header.set_data_dtype(np.uint16)
    sform_code, qform_code = reference.source_codes or (1, 1)
    image.set_sform(affine, code=sform_code or 1)
    image.set_qform(affine, code=qform_code or sform_code or 1)
    # Keep the recognised suffix: nibabel picks the format, and the
    # compression, from the name it is given.
    suffix = "".join(file_path.suffixes[-2:]) if file_path.name.endswith(".nii.gz") else file_path.suffix
    pending = file_path.with_name(f"{file_path.stem}.partial{suffix}")
    try:
        nib.save(image, str(pending))
        os.replace(pending, file_path)
    finally:
        if pending.exists():
            try:
                pending.unlink()
            except OSError:
                pass
    return file_path


# A cerebral microbleed is 2-10 mm across by definition, so a mask longer than
# this is not one, whatever else it is.
MAX_LESION_MM = 10.0

# Radius of the neighbourhood the background threshold is measured over.
# Deliberately *not* the growth cap: one number cannot both decide how far a
# lesion may grow and how local "local background" is.  Tying them together
# meant relaxing the safety cap swallowed a ventricle edge or a different
# tissue, moved the threshold, and changed the measured volume of a lesion
# that had not changed at all.  8 mm is large enough for a stable median on
# this grid (about 2700 voxels) and small enough to stay one tissue.
BACKGROUND_RADIUS_MM = 8.0

# A converged snap walk takes very few steps; the cap only guards against a
# pathological image where the comparison never settles.
_SNAP_MAX_STEPS = 8


def robust_spread(values: np.ndarray) -> tuple[float, float]:
    """A centre and a spread that the lesion itself cannot inflate.

    The threshold for region growing has to describe the *background*, but it
    is measured on a cube that necessarily contains the lesion and whatever
    anatomy sits within the growth cap.  A standard deviation is dominated by
    exactly those outliers, so enlarging the cap tightened the threshold and
    could reject the lesion the reader had just pointed at -- measured on this
    dataset, the volume moved by up to a factor of 44 as the cap changed, and
    two real findings came back as a single voxel.

    The median absolute deviation ignores them: half the sample has to move
    before it does.  The 1.4826 factor makes it match a standard deviation on
    normally distributed data, so ``sensitivity`` keeps its old meaning.
    """

    sample = np.asarray(values, dtype=np.float32).reshape(-1)
    sample = sample[np.isfinite(sample)]
    if sample.size == 0:
        return 0.0, 0.0
    centre = float(np.median(sample))
    spread = float(np.median(np.abs(sample - centre))) * 1.4826
    if spread <= 0:
        # A region flat enough to have no deviation at all: fall back rather
        # than divide the world into "equal to the median" and "not".
        spread = float(np.std(sample))
    return centre, spread


def _fill_enclosed(mask: np.ndarray) -> np.ndarray:
    """Fill voxels that the mask completely surrounds.

    A voxel inside a microbleed that happens to sit above the threshold is
    noise, not background: nothing outside can reach it.  Leaving those holes
    in understates the volume and makes the mask look moth-eaten at lesion
    zoom.
    """

    if not mask.any():
        return mask
    voxels = np.argwhere(mask)
    low = voxels.min(axis=0)
    high = voxels.max(axis=0) + 1
    window = mask[low[0]:high[0], low[1]:high[1], low[2]:high[2]]
    # One voxel of padding all round, so "reachable from outside" is well
    # defined even when the lesion touches the window edge.
    padded = np.zeros(np.asarray(window.shape) + 2, dtype=bool)
    padded[1:-1, 1:-1, 1:-1] = window

    outside = np.zeros(padded.shape, dtype=bool)
    stack: list[tuple[int, int, int]] = [(0, 0, 0)]
    outside[0, 0, 0] = True
    shape = padded.shape
    while stack:
        x, y, z = stack.pop()
        for dx, dy, dz in ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)):
            nx, ny, nz = x + dx, y + dy, z + dz
            if not (0 <= nx < shape[0] and 0 <= ny < shape[1] and 0 <= nz < shape[2]):
                continue
            if outside[nx, ny, nz] or padded[nx, ny, nz]:
                continue
            outside[nx, ny, nz] = True
            stack.append((nx, ny, nz))

    filled = mask.copy()
    holes = ~padded & ~outside
    filled[low[0]:high[0], low[1]:high[1], low[2]:high[2]] |= holes[1:-1, 1:-1, 1:-1]
    return filled


def lesion_shape(
    mask: np.ndarray,
    voxel_sizes: Iterable[float],
    *,
    reached_cap: bool = False,
) -> dict[str, float | bool]:
    """Measure a mask, and say whether it still looks like a microbleed.

    Returned rather than acted on: a mask that ran down a vessel should be
    shown to the reader as unreliable, not silently trimmed to whatever a
    heuristic guesses the lesion part was.  Trimming would invent a boundary;
    reporting lets the reader redraw one.

    ``suspect`` rests on two things that are true by definition rather than by
    calibration: a microbleed is at most 10 mm across, and a region that
    stopped growing because it hit the safety cap stopped for a reason that
    has nothing to do with the image, so its size is the cap's answer and not
    a measurement.  ``elongation`` is reported alongside because it is
    informative, but it does not decide: measured over twenty real findings on
    this dataset, plausible microbleeds run 2.9-6.0, so any threshold on it
    flags almost everything.
    """

    sizes = np.asarray([float(size) for size in voxel_sizes][:3], dtype=np.float64)
    voxels = np.argwhere(np.asarray(mask, dtype=bool))
    count = int(len(voxels))
    result: dict[str, float | bool] = {
        "voxel_count": count,
        "volume_mm3": float(count) * float(np.prod(sizes)),
        "diameter_mm": 0.0,
        "longest_mm": 0.0,
        "elongation": 1.0,
        "reached_cap": bool(reached_cap),
        "suspect": False,
    }
    if count == 0:
        return result

    points = voxels * sizes
    extent = points.max(axis=0) - points.min(axis=0) + sizes
    result["longest_mm"] = float(extent.max())
    result["diameter_mm"] = float(
        2.0 * (3.0 * float(result["volume_mm3"]) / (4.0 * np.pi)) ** (1.0 / 3.0)
    )

    # Second moments rather than the bounding box: a vessel running diagonally
    # fills a cube-shaped box while still being obviously a line.
    if count >= 4:
        centred = points - points.mean(axis=0)
        eigenvalues = np.linalg.eigvalsh(np.cov(centred, rowvar=False))
        eigenvalues = np.clip(eigenvalues, 0.0, None)
        largest = float(eigenvalues.max())
        smallest = float(eigenvalues.min())
        # A voxel grid cannot resolve a spread finer than one voxel.
        floor = (float(sizes.min()) / 2.0) ** 2
        result["elongation"] = float(np.sqrt(max(largest, floor) / max(smallest, floor)))

    result["suspect"] = bool(reached_cap or float(result["longest_mm"]) > MAX_LESION_MM)
    return result


def snap_to_extremum(
    data: np.ndarray,
    voxel: Iterable[float],
    voxel_sizes: Iterable[float],
    *,
    dark: bool,
    radius_mm: float = 2.0,
) -> np.ndarray:
    """Move a point onto the strongest voxel of the focus it is next to.

    A reader marking a 3 mm lesion cannot click its centre by eye, and the
    coordinate they record is the one the analysis uses -- half a millimetre of
    hand tremor is a fifth of a small microbleed.  The darkest voxel on SWI
    (brightest on QSM) within a short reach is a better answer than the pixel
    the mouse happened to be over, and it is reproducible between readers.

    The search radius is deliberately small: this refines a click, it does not
    look for a lesion.  A point over flat tissue comes back unchanged, and so
    does one outside the volume.

    The search repeats from wherever it lands until it stops moving, because a
    single step is not idempotent -- the point it reaches can itself have a
    darker neighbour just outside the first window -- and a snap that gives a
    different answer depending on exactly where the reader clicked is worth
    nothing.  Converged, the result is a genuine local extremum, which is the
    property that makes two readers agree.  Total travel is capped so that
    refining a click can never turn into walking to a different lesion.
    """

    values = np.asarray(data)
    point = np.asarray([float(value) for value in voxel], dtype=np.float64)
    if values.ndim != 3 or point.shape != (3,):
        return point
    sizes = [float(size) for size in voxel_sizes][:3]
    start = np.asarray([int(round(value)) for value in point])
    if not all(0 <= start[axis] < values.shape[axis] for axis in range(3)):
        return point

    budget_sq = (2.0 * float(radius_mm)) ** 2
    scale = np.asarray(sizes, dtype=np.float64)
    current = start
    # One step per iteration; a bounded walk cannot loop, because each step is
    # strictly more extreme than the last.
    for _ in range(_SNAP_MAX_STEPS):
        centre = [int(value) for value in current]
        reach = [max(1, int(np.ceil(radius_mm / max(sizes[axis], 1e-6)))) for axis in range(3)]
        lows = [max(0, centre[axis] - reach[axis]) for axis in range(3)]
        highs = [min(values.shape[axis], centre[axis] + reach[axis] + 1) for axis in range(3)]
        cube = np.asarray(
            values[lows[0]:highs[0], lows[1]:highs[1], lows[2]:highs[2]], dtype=np.float32
        )
        local = [centre[axis] - lows[axis] for axis in range(3)]
        grids = np.ogrid[tuple(slice(0, size) for size in cube.shape)]
        distance_sq = sum(
            ((grid - local[axis]) * sizes[axis]) ** 2 for axis, grid in enumerate(grids)
        )
        reachable = distance_sq <= radius_mm**2
        if not reachable.any():
            break
        candidates = np.where(reachable, cube, np.inf if dark else -np.inf)
        best = np.unravel_index(
            int(np.argmin(candidates) if dark else np.argmax(candidates)), cube.shape
        )
        # Flat tissue: nothing here is more extreme than where we stand, so
        # there is nothing to snap to and moving would be inventing a focus.
        if float(cube[best]) == float(cube[tuple(local)]):
            break
        step = np.asarray([best[axis] + lows[axis] for axis in range(3)])
        if float(np.sum(((step - start) * scale) ** 2)) > budget_sq:
            break
        current = step

    if np.array_equal(current, start):
        return point
    return current.astype(np.float64)



def segment_lesion(
    data: np.ndarray,
    seed: Iterable[float] | np.ndarray,
    voxel_sizes: Iterable[float],
    *,
    dark: bool,
    sensitivity: float = 1.5,
    radius_mm: float = 6.0,
    background_mm: float = BACKGROUND_RADIUS_MM,
) -> tuple[np.ndarray, dict[str, float | bool]]:
    """Grow a lesion mask and measure it, in one step.

    This is what the viewer calls: the measurement is only meaningful next to
    the mask it describes, and whether growth stopped on its own or ran into
    the safety cap is knowable here and nowhere else.
    """

    mask = grow_lesion(
        data, seed, voxel_sizes,
        dark=dark, sensitivity=sensitivity, radius_mm=radius_mm, background_mm=background_mm,
    )
    return mask, lesion_shape(
        mask, voxel_sizes, reached_cap=_touches_cap(mask, seed, voxel_sizes, radius_mm)
    )


def _touches_cap(
    mask: np.ndarray,
    seed: Iterable[float] | np.ndarray,
    voxel_sizes: Iterable[float],
    radius_mm: float,
) -> bool:
    """Did growth stop because of the image, or because of the cap?"""

    voxels = np.argwhere(np.asarray(mask, dtype=bool))
    if voxels.size == 0:
        return False
    sizes = np.asarray([float(size) for size in voxel_sizes][:3], dtype=np.float64)
    seed_array = np.asarray(seed)
    if seed_array.dtype == bool and seed_array.shape == tuple(np.asarray(mask).shape):
        seed_voxels = np.argwhere(seed_array)
        if seed_voxels.size == 0:
            return False
        centre = seed_voxels.mean(axis=0)
    else:
        centre = np.asarray([float(value) for value in seed_array], dtype=np.float64)
    distances = np.linalg.norm((voxels - centre) * sizes, axis=1)
    # Within one voxel of the boundary counts as having reached it.
    return bool(distances.max() >= float(radius_mm) - float(sizes.max()))


def grow_lesion(
    data: np.ndarray,
    seed: Iterable[float] | np.ndarray,
    voxel_sizes: Iterable[float],
    *,
    dark: bool,
    sensitivity: float = 1.5,
    radius_mm: float = 6.0,
    background_mm: float = BACKGROUND_RADIUS_MM,
) -> np.ndarray:
    """Grow a microbleed mask outwards from a seed.

    A microbleed is a small, roughly round focus that is dark on SWI and bright
    on QSM, so a connected region that stays past a threshold set from the local
    background segments it well.  Growth is confined to a sphere (in
    millimetres, so anisotropic voxels behave), which keeps a vessel running
    through the seed from being followed across the whole slab.

    ``seed`` is either one voxel or a boolean mask.  A mask is how a reader
    hands over a scribble: every painted voxel seeds the growth and the sphere
    is measured from their centre, which handles a lesion that is irregular, or
    one where a single point would have landed on an unrepresentative voxel.

    ``radius_mm`` caps how far growth may travel and nothing else;
    ``background_mm`` decides how large a neighbourhood the threshold is
    measured over.  They used to be the same number, which made a safety cap
    silently rescale the answer.

    The result is the connected region reached from the seed, with any voxels
    it completely encloses filled in.  It is *not* trimmed for plausibility --
    pass it to :func:`lesion_shape` to find out whether it still looks like a
    microbleed rather than a length of vessel.

    Returns a boolean mask with the same shape as ``data``.
    """

    values = np.asarray(data)
    if values.ndim != 3:
        raise ValueError("Region growing expects a 3D volume.")
    sizes = [float(size) for size in voxel_sizes][:3]

    seed_array = np.asarray(seed)
    if seed_array.dtype == bool and seed_array.shape == values.shape:
        seed_voxels = np.argwhere(seed_array)
        if seed_voxels.size == 0:
            return np.zeros(values.shape, dtype=bool)
        centre = [int(round(float(value))) for value in seed_voxels.mean(axis=0)]
        # Reach past the scribble itself, not just past its centre.
        margin = (seed_voxels.max(axis=0) - seed_voxels.min(axis=0)) // 2 + 1
    else:
        seed_voxels = np.asarray([[int(round(float(value))) for value in seed_array]])
        centre = list(seed_voxels[0])
        margin = np.zeros(3, dtype=int)
    if not all(0 <= centre[axis] < values.shape[axis] for axis in range(3)):
        return np.zeros(values.shape, dtype=bool)

    # Work inside a cube that contains both spheres: the one growth may fill
    # and the one the background is measured over, whichever reaches further.
    span_mm = max(float(radius_mm), float(background_mm))
    reach = [
        max(1, int(np.ceil(span_mm / max(sizes[axis], 1e-6))) + int(margin[axis]))
        for axis in range(3)
    ]
    lows = [max(0, centre[axis] - reach[axis]) for axis in range(3)]
    highs = [min(values.shape[axis], centre[axis] + reach[axis] + 1) for axis in range(3)]
    cube = np.asarray(
        values[lows[0]:highs[0], lows[1]:highs[1], lows[2]:highs[2]], dtype=np.float32
    )
    local = [centre[axis] - lows[axis] for axis in range(3)]

    # Distance from the seed, in millimetres, on the cube's own grid.  Both the
    # background neighbourhood and the growth cap are measured with it.
    grids = np.ogrid[tuple(slice(0, size) for size in cube.shape)]
    distance_sq = sum(
        ((grid - local[axis]) * sizes[axis]) ** 2 for axis, grid in enumerate(grids)
    )

    # The background is measured over a fixed neighbourhood, never over the
    # whole growth cube: the cap says how far a lesion may grow, not how local
    # "local background" is.
    neighbourhood = cube[distance_sq <= background_mm**2]
    if neighbourhood.size < 64:
        neighbourhood = cube
    background, spread = robust_spread(neighbourhood)
    if spread <= 0:
        return np.zeros(values.shape, dtype=bool)
    limit = background - sensitivity * spread if dark else background + sensitivity * spread
    inside = cube < limit if dark else cube > limit

    # Seeds are inside by definition: the reader pointed at them.
    starts: list[tuple[int, int, int]] = []
    for voxel in seed_voxels:
        point = tuple(int(voxel[axis]) - lows[axis] for axis in range(3))
        if all(0 <= point[axis] < cube.shape[axis] for axis in range(3)):
            inside[point] = True
            starts.append(point)
    if not starts:
        return np.zeros(values.shape, dtype=bool)

    allowed_sq = (radius_mm + float(np.max(margin * np.asarray(sizes)))) ** 2
    inside &= distance_sq <= allowed_sq
    for point in starts:
        inside[point] = True

    # Breadth-first flood fill from the seeds over the thresholded voxels.
    mask = np.zeros(cube.shape, dtype=bool)
    stack = list(starts)
    for point in starts:
        mask[point] = True
    shape = cube.shape
    while stack:
        x, y, z = stack.pop()
        for dx, dy, dz in ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)):
            nx, ny, nz = x + dx, y + dy, z + dz
            if not (0 <= nx < shape[0] and 0 <= ny < shape[1] and 0 <= nz < shape[2]):
                continue
            if mask[nx, ny, nz] or not inside[nx, ny, nz]:
                continue
            mask[nx, ny, nz] = True
            stack.append((nx, ny, nz))

    grown = np.zeros(values.shape, dtype=bool)
    grown[lows[0]:highs[0], lows[1]:highs[1], lows[2]:highs[2]] = _fill_enclosed(mask)
    return grown


def subsample(data: np.ndarray, max_samples: int = 2_000_000) -> np.ndarray:
    """Return a strided view small enough for a fast percentile.

    A 1st/99th percentile of a whole MRI volume is stable under regular
    subsampling, and the strided view avoids the large temporary copies that
    a full-volume percentile allocates on every case load.
    """

    values = np.asarray(data)
    if values.size <= max_samples or values.ndim != 3:
        return values
    step = int(np.ceil((values.size / float(max_samples)) ** (1.0 / 3.0)))
    step = max(1, step)
    return values[::step, ::step, ::step]


def ras_to_voxel(affine: np.ndarray, ras: Iterable[float]) -> np.ndarray:
    point = np.asarray([float(x) for x in ras], dtype=np.float64)
    if point.shape != (3,):
        raise ValueError("RAS must contain exactly three numeric values.")
    homogeneous = np.append(point, 1.0)
    return (np.linalg.inv(np.asarray(affine)) @ homogeneous)[:3]


def voxel_to_ras(affine: np.ndarray, voxel: Iterable[float]) -> np.ndarray:
    point = np.asarray([float(x) for x in voxel], dtype=np.float64)
    if point.shape != (3,):
        raise ValueError("Voxel coordinates must contain exactly three numeric values.")
    homogeneous = np.append(point, 1.0)
    return (np.asarray(affine) @ homogeneous)[:3]


def clamp_voxel(voxel: Iterable[float], shape: tuple[int, ...]) -> np.ndarray:
    point = np.asarray([float(x) for x in voxel], dtype=np.float64)
    limits = np.asarray(shape[:3], dtype=np.float64) - 1
    return np.clip(np.rint(point).astype(int), 0, limits.astype(int))


def voxel_in_bounds(voxel: Iterable[float], shape: tuple[int, ...], tolerance: float = 0.5) -> bool:
    point = np.asarray([float(x) for x in voxel], dtype=np.float64)
    limits = np.asarray(shape[:3], dtype=np.float64)
    return bool(np.all(point >= -tolerance) and np.all(point < limits - tolerance))


def extract_plane(data: np.ndarray, plane: str, index: int) -> tuple[np.ndarray, tuple[str, str]]:
    """Return a 2D image plus the voxel axes represented by its columns/rows."""

    if plane == "axial":
        image = data[:, :, int(index)].T
        return image, ("x", "y")
    if plane == "coronal":
        image = data[:, int(index), :].T
        return image, ("x", "z")
    if plane == "sagittal":
        image = data[int(index), :, :].T
        return image, ("y", "z")
    raise ValueError(f"Unknown plane: {plane}")


def plane_crosshair(voxel: Iterable[float], plane: str) -> tuple[float, float]:
    point = np.asarray([float(x) for x in voxel], dtype=np.float64)
    if plane == "axial":
        return float(point[0]), float(point[1])
    if plane == "coronal":
        return float(point[0]), float(point[2])
    if plane == "sagittal":
        return float(point[1]), float(point[2])
    raise ValueError(f"Unknown plane: {plane}")


def robust_window(image: np.ndarray, fallback: np.ndarray | None = None) -> tuple[float, float]:
    """Return a stable grayscale window for a slice.

    Some MIP or edge slices can be uniform or nearly empty.  A local
    percentile window for such a slice would collapse to a one-value range,
    which Plotly/Canvas can display as a black panel after navigation.  In
    that case use a finite, non-zero window from the whole volume before
    falling back to a small symmetric range.
    """

    values = np.asarray(image, dtype=np.float32)
    finite = values[np.isfinite(values)]
    nonzero = finite[np.abs(finite) > 1e-12]
    sample = nonzero if nonzero.size >= 32 else finite

    def _window(candidate: np.ndarray) -> tuple[float, float] | None:
        candidate = np.asarray(candidate, dtype=np.float32)
        candidate = candidate[np.isfinite(candidate)]
        if candidate.size == 0:
            return None
        nonzero_candidate = candidate[np.abs(candidate) > 1e-12]
        sample_candidate = nonzero_candidate if nonzero_candidate.size >= 32 else candidate
        low_value, high_value = np.percentile(sample_candidate, [1.0, 99.0])
        if not np.isfinite(low_value) or not np.isfinite(high_value):
            return None
        if high_value <= low_value:
            low_value = float(np.min(sample_candidate))
            high_value = float(np.max(sample_candidate))
        if high_value <= low_value:
            return None
        return float(low_value), float(high_value)

    local = _window(sample) if sample.size else None
    if local is not None:
        return local
    if fallback is not None:
        whole_volume = _window(np.asarray(fallback, dtype=np.float32))
        if whole_volume is not None:
            return whole_volume
    # A symmetric fallback keeps an all-zero slice visibly black without
    # producing a zero-width colour range or a renderer error.
    if finite.size:
        center = float(np.nanmean(finite))
        spread = max(abs(float(np.nanmin(finite) - center)), abs(float(np.nanmax(finite) - center)), 1e-6)
        return center - spread, center + spread
    return -1.0, 1.0


# The four corners of a voxel's face at offset 0 along each axis, in voxel
# units.  Winding is not consistent between the three and does not need to be:
# the surface is shaded and culled from the face normal, which comes from the
# axis rather than from the vertex order.
_FACE_CORNERS = (
    np.array([[0, 0, 0], [0, 1, 0], [0, 1, 1], [0, 0, 1]], dtype=np.float64),
    np.array([[0, 0, 0], [0, 0, 1], [1, 0, 1], [1, 0, 0]], dtype=np.float64),
    np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=np.float64),
)


def lesion_surface(
    mask: np.ndarray,
    voxel_sizes: Iterable[float],
    *,
    centre: Iterable[float] | None = None,
    smooth: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """The mask's own surface: every voxel face with nothing on the far side.

    Blocky by default, and that is the honest default: a surface re-derived
    from an interpolated threshold shows a plausible lesion where the reader
    painted a ragged one, and the point of looking at a mask in three
    dimensions is to see what is actually stored -- whether the region grower
    slipped down a vessel, whether a stroke is one slice thick.  It is also
    cheap: a 5 mm microbleed on 0.5 mm voxels is about 500 voxels and 500
    faces, which paints in around two milliseconds.

    ``smooth`` runs that many Laplacian passes over the surface, each moving
    every vertex halfway towards the average of the vertices it shares an edge
    with.  This is a smoothing *of the stored surface*, not a different
    isosurface: no vertex is added, removed or re-thresholded, so the shape
    stays a rendering of the same voxels rather than a guess at what they
    might have been.

    Laplacian smoothing shrinks, and how much depends on the shape.  Measured
    as the mean vertex distance from the centre: a 5 mm ball loses 0.7% a
    pass, a one-voxel-thick slab 4.5% -- the flatter the thing, the more it
    pulls in.  Nothing measured is taken from this mesh, so the shrinkage is
    cosmetic: volume, diameter and voxel count all come from the voxels
    themselves via ``lesion_shape``.

    ``centre`` is the millimetre point the coordinates are measured from; it
    defaults to the centre of the mask's bounding box, so a caller can rotate
    the lesion about itself without knowing where in the volume it sat.  Pass
    the volume's own centre to place the lesion in the brain instead.

    Returns ``(quads, normals)``: quads is ``(n, 4, 3)`` of millimetre
    coordinates, normals ``(n, 3)`` of unit vectors pointing out of the
    surface, in the same millimetre space.
    """

    mask = np.asarray(mask, dtype=bool)
    spacing = np.asarray([float(size) for size in voxel_sizes][:3], dtype=np.float64)
    empty = (np.zeros((0, 4, 3), dtype=np.float64), np.zeros((0, 3), dtype=np.float64))
    if mask.ndim != 3 or not mask.any():
        return empty

    # Work on the bounding box, not the whole 256^3 volume: a microbleed is a
    # few hundred voxels of it, and the six shifted comparisons below would
    # otherwise touch sixteen million each.
    coords = np.argwhere(mask)
    low = coords.min(axis=0)
    high = coords.max(axis=0) + 1
    sub = mask[low[0]:high[0], low[1]:high[1], low[2]:high[2]]
    padded = np.pad(sub, 1)

    quads: list[np.ndarray] = []
    normals: list[np.ndarray] = []
    for axis in range(3):
        for side in (0, 1):
            # ``roll`` by +1 brings the i-1 neighbour into place, by -1 the
            # i+1 one; the pad makes the volume's own boundary count as empty,
            # which is what makes a mask touching the edge still closed.
            neighbour = np.roll(padded, 1 if side == 0 else -1, axis=axis)[1:-1, 1:-1, 1:-1]
            exposed = np.argwhere(sub & ~neighbour)
            if not len(exposed):
                continue
            corners = _FACE_CORNERS[axis].copy()
            corners[:, axis] += side
            quads.append((exposed[:, None, :] + corners[None, :, :] + low) * spacing)
            direction = np.zeros(3)
            direction[axis] = 1.0 if side else -1.0
            normals.append(np.repeat(direction[None, :], len(exposed), axis=0))

    if not quads:
        return empty
    vertices = np.concatenate(quads)
    origin = (
        np.asarray([float(value) for value in centre][:3], dtype=np.float64)
        if centre is not None
        else (low + high) / 2.0 * spacing
    )
    vertices = vertices - origin
    face_normals = np.concatenate(normals)
    if smooth > 0:
        vertices = _smooth_quad_mesh(vertices, int(smooth))
        face_normals = _quad_normals(vertices, face_normals)
    return vertices, face_normals


def _quad_normals(quads: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    """Geometric normals, for a surface whose faces are no longer axis-aligned.

    Kept pointing the same way as the axis normals they replace: the cross
    product of the diagonals is perpendicular to the face but its sign depends
    on the winding, which is not consistent between the three face
    orientations and does not need to be for anything else.
    """

    if not len(quads):
        return fallback
    normals = np.cross(quads[:, 2] - quads[:, 0], quads[:, 3] - quads[:, 1])
    lengths = np.linalg.norm(normals, axis=1)
    usable = lengths > 1e-12
    normals[usable] /= lengths[usable, None]
    normals[~usable] = fallback[~usable]
    flipped = np.einsum("ij,ij->i", normals, fallback) < 0
    normals[flipped] *= -1.0
    return normals


def _smooth_quad_mesh(quads: np.ndarray, passes: int) -> np.ndarray:
    """Laplacian smoothing over the vertices a quad soup shares.

    The faces arrive as independent quads, so the first job is to find which
    of their corners are the same point.  Corners land on exact voxel-corner
    multiples of the spacing, so rounding to a fine grid welds them without
    the tolerance guesswork a general mesh would need.
    """

    flat = quads.reshape(-1, 3)
    keys = np.round(flat, 6)
    _unique, index, inverse = np.unique(keys, axis=0, return_index=True, return_inverse=True)
    points = flat[index].copy()
    inverse = inverse.reshape(-1, 4)

    # Every edge of every quad, both ways, so the average below is symmetric.
    edges = np.concatenate(
        [np.stack([inverse[:, a], inverse[:, (a + 1) % 4]], axis=1) for a in range(4)]
    )
    edges = np.concatenate([edges, edges[:, ::-1]])
    counts = np.bincount(edges[:, 0], minlength=len(points)).astype(np.float64)
    counts[counts == 0] = 1.0

    for _ in range(max(0, passes)):
        summed = np.zeros_like(points)
        np.add.at(summed, edges[:, 0], points[edges[:, 1]])
        points = 0.5 * points + 0.5 * (summed / counts[:, None])
    return points[inverse]


# How many samples along the longest side of the resampled cube the brain
# context is projected from.  Measured on this machine: a nearest-neighbour
# resample plus projection costs 14.5 ms at 96, 34.8 ms at 128 and 67.5 ms at
# 160, and the cube is projected twice a frame (once for the half behind the
# lesion, once for the half in front).  128 keeps a drag above 20 fps while
# still resolving a 3 mm lesion's neighbourhood.
CONTEXT_CUBE_SIZE = 128


def isotropic_context(
    data: np.ndarray, voxel_sizes: Iterable[float], size: int = CONTEXT_CUBE_SIZE
) -> tuple[np.ndarray, float]:
    """Resample a volume into a cube of equal-sided voxels, once.

    Rotating an anisotropic array by index would shear the anatomy -- the SWI
    here is 0.5 x 0.5 x 1.0 mm, so a head turned forty-five degrees would come
    out a third taller than it is.  Doing it once at load time means every
    later frame is a plain index rotation.

    Nearest neighbour rather than trilinear: this is a backdrop drawn at a
    fraction of the canvas size and then scaled up, so interpolation would
    cost a factor of eight in samples to be thrown away by the scaling.  The
    lesion surface, which is the thing being measured, does not come from
    here.

    Returns ``(cube, mm_per_voxel)``.  The cube is centred on the volume's own
    centre, so millimetre coordinates measured from that centre index it
    directly.
    """

    data = np.asarray(data)
    spacing = np.asarray([float(value) for value in voxel_sizes][:3], dtype=np.float64)
    if data.ndim != 3 or not data.size:
        return np.zeros((0, 0, 0), dtype=np.float32), 1.0
    extent = np.asarray(data.shape, dtype=np.float64) * spacing
    size = max(8, int(size))
    mm_per_voxel = float(extent.max() / size)

    # One axis of the cube per volume axis, in millimetres from the centre.
    axes = []
    for length, step, count in zip(extent, spacing, data.shape):
        samples = np.arange(size, dtype=np.float64) - (size - 1) / 2.0
        millimetres = samples * mm_per_voxel + length / 2.0
        index = np.clip(np.rint(millimetres / step - 0.5), 0, count - 1).astype(np.intp)
        axes.append(index)
    cube = data[np.ix_(axes[0], axes[1], axes[2])]
    return np.ascontiguousarray(cube, dtype=np.float32), mm_per_voxel


def project_context(
    cube: np.ndarray,
    rotation: np.ndarray,
    *,
    near: float | None = None,
    far: float | None = None,
) -> np.ndarray:
    """Mean intensity through a rotated cube, along the viewing direction.

    A mean rather than a maximum: a maximum projection of SWI is all scalp and
    skull, which is the one part of the picture nobody is looking for.  The
    mean keeps the head's silhouette and the ventricles, which is what makes a
    lesion's position readable.

    ``near`` and ``far`` bound the slab in view depth, in cube voxels from the
    centre, so a caller can draw what is behind the lesion, then the lesion,
    then what is in front of it -- the cheapest way to get the occlusion right
    without sorting anything, and it costs nothing: the two halves together
    sample exactly the voxels one full projection would.

    Returned indexed ``[row, column]``, ready for a screen.  Timed on this
    machine with the grid cached: 22.7 ms a frame for both halves of a 96
    cube, 54.8 ms for a 128 one.
    """

    cube = np.asarray(cube, dtype=np.float32)
    if cube.ndim != 3 or not cube.size:
        return np.zeros((0, 0), dtype=np.float32)
    size = cube.shape[0]
    half = np.float32((size - 1) / 2.0)
    line = np.arange(size, dtype=np.float32) - half
    start, stop = 0, size
    if near is not None:
        start = int(np.searchsorted(line, float(near), side="left"))
    if far is not None:
        stop = int(np.searchsorted(line, float(far), side="right"))
    if stop <= start:
        return np.zeros((size, size), dtype=np.float32)

    # View grid -> world, one component at a time by broadcasting three 1-D
    # axes.  Materialising the grid and doing a matmul gave the same numbers
    # to the bit, but held a 25 MB array per cube size and ran 1.6x slower
    # (58.3 ms against 36.4 for a 128 cube, both depth halves).
    #
    # The rotation is orthonormal, so its transpose is its inverse.  Axes are
    # laid out (depth, row, column) with each point's components in (x, y,
    # depth) order -- the same order the surface renderer rotates its vertices
    # into, so one matrix serves both.
    inverse = np.asarray(rotation, dtype=np.float32).T
    depth = line[start:stop, None, None]
    rows = line[None, :, None]
    columns = line[None, None, :]
    index = []
    for axis in range(3):
        position = (
            inverse[axis, 0] * columns + inverse[axis, 1] * rows + inverse[axis, 2] * depth
        )
        np.add(position, half, out=position)
        np.rint(position, out=position)
        index.append(position.astype(np.int32))
    inside = np.ones(index[0].shape, dtype=bool)
    for axis in range(3):
        inside &= (index[axis] >= 0) & (index[axis] < size)
        np.clip(index[axis], 0, size - 1, out=index[axis])
    sampled = np.where(inside, cube[index[0], index[1], index[2]], np.float32(0.0))
    return sampled.mean(axis=0)
