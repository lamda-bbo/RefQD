from ._numpy_head import Any, Callable, DType, DTypeLike, DimSize, EllipsisType, Optional, PadValueLike, PrecisionLike, Sequence, Shape, Tree, TreeT, Union, dtype, jnp, lax, np, typing  # noqa: E501


def abs(x: TreeT, /) -> TreeT:  # noqa: E501
    ...


def absolute(x: TreeT, /) -> TreeT:  # noqa: E501
    ...


def add(x1: TreeT, x2: TreeT, /) -> TreeT:  # noqa: E501
    ...


def all(a: TreeT, axis: Tree = None, out: None = None, keepdims: bool = False, *, where: Tree = None) -> TreeT:  # noqa: E501
    ...


def allclose(a: TreeT, b: Tree, rtol: Tree = 1e-05, atol: Tree = 1e-08, equal_nan: bool = False) -> TreeT:  # noqa: E501
    ...


def amax(a: TreeT, axis: Tree = None, out: None = None, keepdims: bool = False, initial: Tree = None, where: Tree = None) -> TreeT:  # noqa: E501
    ...


def amin(a: TreeT, axis: Tree = None, out: None = None, keepdims: bool = False, initial: Tree = None, where: Tree = None) -> TreeT:  # noqa: E501
    ...


def angle(z: TreeT, deg: bool = False) -> TreeT:  # noqa: E501
    ...


def any(a: TreeT, axis: Tree = None, out: None = None, keepdims: bool = False, *, where: Tree = None) -> TreeT:  # noqa: E501
    ...


def append(arr: TreeT, values: Tree, axis: int | None = None) -> TreeT:  # noqa: E501
    ...


def apply_along_axis(func1d: Callable, axis: int, arr: TreeT, *args, **kwargs) -> TreeT:  # noqa: E501
    ...


def apply_over_axes(func: Callable[[TreeT, int], TreeT], a: Tree, axes: Sequence[int]) -> TreeT:  # noqa: E501
    ...


def arange(start: DimSize, stop: Optional[DimSize] = None, step: DimSize | None = None, dtype: DTypeLike | None = None) -> Tree:  # noqa: E501
    ...


def arccos(x: TreeT, /) -> TreeT:  # noqa: E501
    ...


def arccosh(x: TreeT, /) -> TreeT:  # noqa: E501
    ...


def arcsin(x: TreeT, /) -> TreeT:  # noqa: E501
    ...


def arcsinh(x: TreeT, /) -> TreeT:  # noqa: E501
    ...


def arctan(x: TreeT, /) -> TreeT:  # noqa: E501
    ...


def arctan2(x1: TreeT, x2: TreeT, /) -> TreeT:  # noqa: E501
    ...


def arctanh(x: TreeT, /) -> TreeT:  # noqa: E501
    ...


def argmax(a: TreeT, axis: int | None = None, out: None = None, keepdims: bool | None = None) -> TreeT:  # noqa: E501
    ...


def argmin(a: TreeT, axis: int | None = None, out: None = None, keepdims: bool | None = None) -> TreeT:  # noqa: E501
    ...


def argpartition(a: TreeT, kth: int, axis: int = -1) -> TreeT:  # noqa: E501
    ...


def argsort(a: TreeT, axis: int | None = -1, kind: str = 'stable', order: None = None) -> TreeT:  # noqa: E501
    ...


def argwhere(a: TreeT, *, size: int | None = None, fill_value: Tree | None = None) -> TreeT:  # noqa: E501
    ...


def around(a: TreeT, decimals: int = 0, out: None = None) -> TreeT:  # noqa: E501
    ...


def array(object: Any, dtype: DTypeLike | None = None, copy: bool = True, order: str | None = 'K', ndmin: int = 0) -> Tree:  # noqa: E501
    ...


def array_equal(a1: TreeT, a2: Tree, equal_nan: bool = False) -> TreeT:  # noqa: E501
    ...


def array_equiv(a1: TreeT, a2: Tree) -> TreeT:  # noqa: E501
    ...


def array_split(ary: TreeT, indices_or_sections: int | Sequence[int] | Tree, axis: int = 0) -> list[TreeT]:  # noqa: E501
    ...


def asarray(a: Any, dtype: DTypeLike | None = None, order: str | None = None) -> Tree:  # noqa: E501
    ...


def asis(a: TreeT) -> TreeT:  # noqa: E501
    ...


def atleast_1d(*arys: TreeT) -> TreeT | list[TreeT]:  # noqa: E501
    ...


def atleast_2d(*arys: TreeT) -> TreeT | list[TreeT]:  # noqa: E501
    ...


def atleast_3d(*arys: TreeT) -> TreeT | list[TreeT]:  # noqa: E501
    ...


def average(a: TreeT, axis: Tree = None, weights: Tree = None, returned: bool = False, keepdims: bool = False) -> TreeT:  # noqa: E501
    ...


def bartlett(M: int) -> Tree:  # noqa: E501
    ...


def bfloat16(x: Any) -> Tree:  # noqa: E501
    ...


def bincount(x: TreeT, weights: Tree | None = None, minlength: int = 0, *, length: int | None = None) -> TreeT:  # noqa: E501
    ...


def bitwise_and(x1: TreeT, x2: TreeT, /) -> TreeT:  # noqa: E501
    ...


def bitwise_count(x: TreeT, /) -> TreeT:  # noqa: E501
    ...


def bitwise_not(x: TreeT, /) -> TreeT:  # noqa: E501
    ...


def bitwise_or(x1: TreeT, x2: TreeT, /) -> TreeT:  # noqa: E501
    ...


def bitwise_xor(x1: TreeT, x2: TreeT, /) -> TreeT:  # noqa: E501
    ...


def blackman(M: int) -> Tree:  # noqa: E501
    ...


def block(arrays: TreeT | list[TreeT]) -> TreeT:  # noqa: E501
    ...


def bool_(x: Any) -> Tree:  # noqa: E501
    ...


def broadcast_arrays(*args: TreeT) -> list[TreeT]:  # noqa: E501
    ...


def broadcast_to(array: TreeT, shape: DimSize | Shape) -> TreeT:  # noqa: E501
    ...


def cbrt(x: TreeT, /) -> TreeT:  # noqa: E501
    ...


def cdouble(x: Any) -> Tree:  # noqa: E501
    ...


def ceil(x: TreeT, /) -> TreeT:  # noqa: E501
    ...


def choose(a: TreeT, choices: Sequence[Tree], out: None = None, mode: str = 'raise') -> TreeT:  # noqa: E501
    ...


def clip(a: TreeT, a_min: Tree | None = None, a_max: Tree | None = None, out: None = None) -> TreeT:  # noqa: E501
    ...


def column_stack(*tup: TreeT) -> TreeT:  # noqa: E501
    ...


def complex128(x: Any) -> Tree:  # noqa: E501
    ...


def complex64(x: Any) -> Tree:  # noqa: E501
    ...


def complex_(x: Any) -> Tree:  # noqa: E501
    ...


def compress(condition: Tree, a: TreeT, axis: int | None = None, out: None = None) -> TreeT:  # noqa: E501
    ...


def concatenate(*arrays: TreeT, axis: int | None = 0, dtype: DTypeLike | None = None) -> TreeT:  # noqa: E501
    ...


def conj(x: TreeT, /) -> TreeT:  # noqa: E501
    ...


def conjugate(x: TreeT, /) -> TreeT:  # noqa: E501
    ...


def convolve(a: TreeT, v: Tree, mode: str = 'full', *, precision: PrecisionLike = None, preferred_element_type: dtype | None = None) -> TreeT:  # noqa: E501
    ...


def copy(a: TreeT, order: str | None = None) -> TreeT:  # noqa: E501
    ...


def copysign(x1: TreeT, x2: TreeT, /) -> TreeT:  # noqa: E501
    ...


def corrcoef(x: TreeT, y: Tree | None = None, rowvar: bool = True) -> TreeT:  # noqa: E501
    ...


def correlate(a: TreeT, v: Tree, mode: str = 'valid', *, precision: PrecisionLike = None, preferred_element_type: dtype | None = None) -> TreeT:  # noqa: E501
    ...


def cos(x: TreeT, /) -> TreeT:  # noqa: E501
    ...


def cosh(x: TreeT, /) -> TreeT:  # noqa: E501
    ...


def count_nonzero(a: TreeT, axis: Tree = None, keepdims: bool = False) -> TreeT:  # noqa: E501
    ...


def cov(m: TreeT, y: Tree | None = None, rowvar: bool = True, bias: bool = False, ddof: int | None = None, fweights: Tree | None = None, aweights: Tree | None = None) -> TreeT:  # noqa: E501
    ...


def csingle(x: Any) -> Tree:  # noqa: E501
    ...


def cumprod(a: TreeT, axis: Tree = None, dtype: Tree = None, out: None = None) -> TreeT:  # noqa: E501
    ...


def cumsum(a: TreeT, axis: Tree = None, dtype: Tree = None, out: None = None) -> TreeT:  # noqa: E501
    ...


def deg2rad(x: TreeT, /) -> TreeT:  # noqa: E501
    ...


def degrees(x: TreeT, /) -> TreeT:  # noqa: E501
    ...


def delete(arr: TreeT, obj: Tree | slice, axis: int | None = None, *, assume_unique_indices: bool = False) -> TreeT:  # noqa: E501
    ...


def diag(v: TreeT, k: int = 0) -> TreeT:  # noqa: E501
    ...


def diag_indices(n: int, ndim: int = 2) -> tuple[Tree, ...]:  # noqa: E501
    ...


def diag_indices_from(arr: TreeT) -> tuple[TreeT, ...]:  # noqa: E501
    ...


def diagflat(v: TreeT, k: int = 0) -> TreeT:  # noqa: E501
    ...


def diagonal(a: TreeT, offset: int = 0, axis1: int = 0, axis2: int = 1) -> TreeT:  # noqa: E501
    ...


def diff(a: TreeT, n: int = 1, axis: int = -1, prepend: Tree | None = None, append: Tree | None = None) -> TreeT:  # noqa: E501
    ...


def digitize(x: TreeT, bins: Tree, right: bool = False) -> TreeT:  # noqa: E501
    ...


def distance(a: TreeT, b: Tree) -> TreeT:  # noqa: E501
    ...


def divide(x1: TreeT, x2: TreeT, /) -> TreeT:  # noqa: E501
    ...


def divmod(x1: TreeT, x2: TreeT, /) -> tuple[TreeT, TreeT]:  # noqa: E501
    ...


def dot(a: TreeT, b: Tree, *, precision: PrecisionLike = None, preferred_element_type: DTypeLike | None = None) -> TreeT:  # noqa: E501
    ...


def double(x: Any) -> Tree:  # noqa: E501
    ...


def dsplit(ary: TreeT, indices_or_sections: int | Sequence[int] | Tree) -> list[TreeT]:  # noqa: E501
    ...


def dstack(*tup: TreeT, dtype: DTypeLike | None = None) -> TreeT:  # noqa: E501
    ...


def duplicate(a: TreeT, repeats: int, axis: int = 0) -> TreeT:  # noqa: E501
    ...


def ediff1d(ary: TreeT, to_end: Tree | None = None, to_begin: Tree | None = None) -> TreeT:  # noqa: E501
    ...


def einsum(subscripts, /, *operands, out: None = None, optimize: str = 'optimal', precision: PrecisionLike = None, preferred_element_type: DTypeLike | None = None, _use_xeinsum: bool = False, _dot_general: Callable[..., TreeT] = lax.dot_general) -> TreeT:  # noqa: E501
    ...


def empty(shape: Any, dtype: DTypeLike | None = None) -> Tree:  # noqa: E501
    ...


def empty_like(prototype: TreeT, dtype: DTypeLike | None = None, shape: Any = None) -> TreeT:  # noqa: E501
    ...


def equal(x1: TreeT, x2: TreeT, /) -> TreeT:  # noqa: E501
    ...


def exp(x: TreeT, /) -> TreeT:  # noqa: E501
    ...


def exp2(x: TreeT, /) -> TreeT:  # noqa: E501
    ...


def expand_dims(a: TreeT, axis: int | Sequence[int]) -> TreeT:  # noqa: E501
    ...


def expm1(x: TreeT, /) -> TreeT:  # noqa: E501
    ...


def extract(condition: Tree, arr: TreeT) -> TreeT:  # noqa: E501
    ...


def eye(N: DimSize, M: DimSize | None = None, k: int = 0, dtype: DTypeLike | None = None) -> Tree:  # noqa: E501
    ...


def fabs(x: TreeT, /) -> TreeT:  # noqa: E501
    ...


def fill_diagonal(a: TreeT, val: Tree, wrap: bool = False, *, inplace: bool = True) -> TreeT:  # noqa: E501
    ...


def fix(x: TreeT, out: None = None) -> TreeT:  # noqa: E501
    ...


def flatnonzero(a: TreeT, *, size: int | None = None, fill_value: None | Tree | tuple[Tree] = None) -> TreeT:  # noqa: E501
    ...


def flip(m: TreeT, axis: int | Sequence[int] | None = None) -> TreeT:  # noqa: E501
    ...


def fliplr(m: TreeT) -> TreeT:  # noqa: E501
    ...


def flipud(m: TreeT) -> TreeT:  # noqa: E501
    ...


def float16(x: Any) -> Tree:  # noqa: E501
    ...


def float32(x: Any) -> Tree:  # noqa: E501
    ...


def float64(x: Any) -> Tree:  # noqa: E501
    ...


def float8_e4m3b11fnuz(x: Any) -> Tree:  # noqa: E501
    ...


def float8_e4m3fn(x: Any) -> Tree:  # noqa: E501
    ...


def float8_e4m3fnuz(x: Any) -> Tree:  # noqa: E501
    ...


def float8_e5m2(x: Any) -> Tree:  # noqa: E501
    ...


def float8_e5m2fnuz(x: Any) -> Tree:  # noqa: E501
    ...


def float_(x: Any) -> Tree:  # noqa: E501
    ...


def float_power(x1: TreeT, x2: TreeT, /) -> TreeT:  # noqa: E501
    ...


def floor(x: TreeT, /) -> TreeT:  # noqa: E501
    ...


def floor_divide(x1: TreeT, x2: TreeT, /) -> TreeT:  # noqa: E501
    ...


def fmax(x1: TreeT, x2: Tree) -> TreeT:  # noqa: E501
    ...


def fmin(x1: TreeT, x2: Tree) -> TreeT:  # noqa: E501
    ...


def fmod(x1: TreeT, x2: TreeT, /) -> TreeT:  # noqa: E501
    ...


def frexp(x: TreeT, /) -> tuple[TreeT, TreeT]:  # noqa: E501
    ...


def from_dlpack(x: Any) -> Tree:  # noqa: E501
    ...


def frombuffer(buffer: bytes | Any, dtype: DTypeLike = float, count: int = -1, offset: int = 0) -> Tree:  # noqa: E501
    ...


def fromfunction(function: Callable[..., TreeT], shape: Any, *, dtype: DTypeLike = float, **kwargs) -> TreeT:  # noqa: E501
    ...


def fromstring(string: str, dtype: DTypeLike = float, count: int = -1, *, sep: str) -> Tree:  # noqa: E501
    ...


def full(shape: Any, fill_value: TreeT, dtype: DTypeLike | None = None) -> TreeT:  # noqa: E501
    ...


def full_like(a: TreeT, fill_value: Tree, dtype: DTypeLike | None = None, shape: Any = None) -> TreeT:  # noqa: E501
    ...


def gcd(x1: TreeT, x2: Tree) -> TreeT:  # noqa: E501
    ...


def geomspace(start: TreeT, stop: Tree, num: int = 50, endpoint: bool = True, dtype: DTypeLike | None = None, axis: int = 0) -> TreeT:  # noqa: E501
    ...


def getitem(a: TreeT, indices: None | int | slice | EllipsisType | Tree | tuple[None | int | slice | EllipsisType | Tree, ...]) -> TreeT:  # noqa: E501
    ...


def gradient(f: TreeT, *varargs: Tree, axis: int | Sequence[int] | None = None, edge_order: int | None = None) -> TreeT | list[TreeT]:  # noqa: E501
    ...


def greater(x1: TreeT, x2: TreeT, /) -> TreeT:  # noqa: E501
    ...


def greater_equal(x1: TreeT, x2: TreeT, /) -> TreeT:  # noqa: E501
    ...


def hamming(M: int) -> Tree:  # noqa: E501
    ...


def hanning(M: int) -> Tree:  # noqa: E501
    ...


def heaviside(x1: TreeT, x2: TreeT, /) -> TreeT:  # noqa: E501
    ...


def histogram(a: TreeT, bins: Tree = 10, range: Sequence[Tree] | None = None, weights: Tree | None = None, density: bool | None = None) -> tuple[TreeT, TreeT]:  # noqa: E501
    ...


def histogram2d(x: TreeT, y: Tree, bins: Tree | list[Tree] = 10, range: Sequence[None | Tree | Sequence[Tree]] | None = None, weights: Tree | None = None, density: bool | None = None) -> tuple[TreeT, TreeT, TreeT]:  # noqa: E501
    ...


def histogram_bin_edges(a: TreeT, bins: Tree = 10, range: None | Tree | Sequence[Tree] = None, weights: Tree | None = None) -> TreeT:  # noqa: E501
    ...


def histogramdd(sample: TreeT, bins: Tree | list[Tree] = 10, range: Sequence[None | Tree | Sequence[Tree]] | None = None, weights: Tree | None = None, density: bool | None = None) -> tuple[TreeT, list[TreeT]]:  # noqa: E501
    ...


def hsplit(ary: TreeT, indices_or_sections: int | Sequence[int] | Tree) -> list[TreeT]:  # noqa: E501
    ...


def hstack(*tup: TreeT, dtype: DTypeLike | None = None) -> TreeT:  # noqa: E501
    ...


def hypot(x1: TreeT, x2: TreeT, /) -> TreeT:  # noqa: E501
    ...


def i0(x: TreeT) -> TreeT:  # noqa: E501
    ...


def identity(n: DimSize, dtype: DTypeLike | None = None) -> Tree:  # noqa: E501
    ...


def imag(val: TreeT, /) -> TreeT:  # noqa: E501
    ...


def indices(dimensions: Sequence[int], dtype: DTypeLike = jnp.int32, sparse: bool = False) -> TreeT | tuple[TreeT, ...]:  # noqa: E501
    ...


def inner(a: TreeT, b: Tree, *, precision: PrecisionLike = None, preferred_element_type: DType | None = None) -> TreeT:  # noqa: E501
    ...


def insert(arr: TreeT, obj: Tree | slice, values: Tree, axis: int | None = None) -> TreeT:  # noqa: E501
    ...


def int16(x: Any) -> Tree:  # noqa: E501
    ...


def int32(x: Any) -> Tree:  # noqa: E501
    ...


def int4(x: Any) -> Tree:  # noqa: E501
    ...


def int64(x: Any) -> Tree:  # noqa: E501
    ...


def int8(x: Any) -> Tree:  # noqa: E501
    ...


def int_(x: Any) -> Tree:  # noqa: E501
    ...


def interp(x: TreeT, xp: Tree, fp: Tree, left: Tree | str | None = None, right: Tree | str | None = None, period: Tree | None = None) -> TreeT:  # noqa: E501
    ...


def intersect1d(ar1: TreeT, ar2: Tree, assume_unique: bool = False, return_indices: bool = False) -> TreeT:  # noqa: E501
    ...


def invert(x: TreeT, /) -> TreeT:  # noqa: E501
    ...


def isclose(a: TreeT, b: Tree, rtol: Tree = 1e-05, atol: Tree = 1e-08, equal_nan: bool = False) -> TreeT:  # noqa: E501
    ...


def iscomplex(x: TreeT) -> TreeT:  # noqa: E501
    ...


def isfinite(x: TreeT, /) -> TreeT:  # noqa: E501
    ...


def isin(element: TreeT, test_elements: Tree, assume_unique: bool = False, invert: bool = False) -> TreeT:  # noqa: E501
    ...


def isinf(x: TreeT, /) -> TreeT:  # noqa: E501
    ...


def isnan(x: TreeT, /) -> TreeT:  # noqa: E501
    ...


def isneginf(x: TreeT, /, out: None = None) -> TreeT:  # noqa: E501
    ...


def isposinf(x: TreeT, /, out: None = None) -> TreeT:  # noqa: E501
    ...


def isreal(x: TreeT) -> TreeT:  # noqa: E501
    ...


def ix_(*args: TreeT) -> tuple[TreeT, ...]:  # noqa: E501
    ...


def kaiser(M: int, beta: TreeT) -> TreeT:  # noqa: E501
    ...


def kron(a: TreeT, b: Tree) -> TreeT:  # noqa: E501
    ...


def lcm(x1: TreeT, x2: Tree) -> TreeT:  # noqa: E501
    ...


def ldexp(x1: TreeT, x2: TreeT, /) -> TreeT:  # noqa: E501
    ...


def left_shift(x1: TreeT, x2: TreeT, /) -> TreeT:  # noqa: E501
    ...


def less(x1: TreeT, x2: TreeT, /) -> TreeT:  # noqa: E501
    ...


def less_equal(x1: TreeT, x2: TreeT, /) -> TreeT:  # noqa: E501
    ...


def lexsort(keys: Union[TreeT, np.ndarray, Sequence[TreeT]], axis: int = -1) -> TreeT:  # noqa: E501
    ...


def linspace(start: TreeT, stop: Tree, num: int = 50, endpoint: bool = True, retstep: bool = False, dtype: DTypeLike | None = None, axis: int = 0) -> TreeT | tuple[TreeT, TreeT]:  # noqa: E501
    ...


def load(*args: Any, **kwargs: Any) -> Tree:  # noqa: E501
    ...


def log(x: TreeT, /) -> TreeT:  # noqa: E501
    ...


def log10(x: TreeT, /) -> TreeT:  # noqa: E501
    ...


def log1p(x: TreeT, /) -> TreeT:  # noqa: E501
    ...


def log2(x: TreeT, /) -> TreeT:  # noqa: E501
    ...


def logaddexp(x1: TreeT, x2: TreeT, /) -> TreeT:  # noqa: E501
    ...


def logaddexp2(x1: TreeT, x2: TreeT, /) -> TreeT:  # noqa: E501
    ...


def logical_and(*args: TreeT) -> TreeT:  # noqa: E501
    ...


def logical_not(*args: TreeT) -> TreeT:  # noqa: E501
    ...


def logical_or(*args: TreeT) -> TreeT:  # noqa: E501
    ...


def logical_xor(*args: TreeT) -> TreeT:  # noqa: E501
    ...


def logspace(start: TreeT, stop: Tree, num: int = 50, endpoint: bool = True, base: Tree = 10.0, dtype: DTypeLike | None = None, axis: int = 0) -> TreeT:  # noqa: E501
    ...


def matmul(a: TreeT, b: Tree, *, precision: PrecisionLike = None, preferred_element_type: DTypeLike | None = None) -> TreeT:  # noqa: E501
    ...


def matrix_transpose(x: TreeT, /) -> TreeT:  # noqa: E501
    ...


def max(a: TreeT, axis: Tree = None, out: None = None, keepdims: bool = False, initial: Tree = None, where: Tree = None) -> TreeT:  # noqa: E501
    ...


def maximum(x1: TreeT, x2: TreeT, /) -> TreeT:  # noqa: E501
    ...


def mean(a: TreeT, axis: Tree = None, dtype: Tree = None, out: None = None, keepdims: bool = False, *, where: Tree = None) -> TreeT:  # noqa: E501
    ...


def median(a: TreeT, axis: Tree = None, out: None = None, overwrite_input: bool = False, keepdims: bool = False) -> TreeT:  # noqa: E501
    ...


def meshgrid(*xi: TreeT, copy: bool = True, sparse: bool = False, indexing: str = 'xy') -> list[TreeT]:  # noqa: E501
    ...


def min(a: TreeT, axis: Tree = None, out: None = None, keepdims: bool = False, initial: Tree = None, where: Tree = None) -> TreeT:  # noqa: E501
    ...


def minimum(x1: TreeT, x2: TreeT, /) -> TreeT:  # noqa: E501
    ...


def mod(x1: TreeT, x2: TreeT, /) -> TreeT:  # noqa: E501
    ...


def modf(x: TreeT, /, out: None = None) -> tuple[TreeT, TreeT]:  # noqa: E501
    ...


def moveaxis(a: TreeT, source: int | Sequence[int], destination: int | Sequence[int]) -> TreeT:  # noqa: E501
    ...


def multiply(x1: TreeT, x2: TreeT, /) -> TreeT:  # noqa: E501
    ...


def nan_to_num(x: TreeT, copy: bool = True, nan: Tree = 0.0, posinf: Tree | None = None, neginf: Tree | None = None) -> TreeT:  # noqa: E501
    ...


def nanargmax(a: TreeT, axis: int | None = None, out: None = None, keepdims: bool | None = None) -> TreeT:  # noqa: E501
    ...


def nanargmin(a: TreeT, axis: int | None = None, out: None = None, keepdims: bool | None = None) -> TreeT:  # noqa: E501
    ...


def nancumprod(a: TreeT, axis: Tree = None, dtype: Tree = None, out: None = None) -> TreeT:  # noqa: E501
    ...


def nancumsum(a: TreeT, axis: Tree = None, dtype: Tree = None, out: None = None) -> TreeT:  # noqa: E501
    ...


def nanmax(a: TreeT, axis: Tree = None, out: None = None, keepdims: bool = False, initial: Tree = None, where: Tree = None) -> TreeT:  # noqa: E501
    ...


def nanmean(a: TreeT, axis: Tree = None, dtype: Tree = None, out: None = None, keepdims: bool = False, where: Tree = None) -> TreeT:  # noqa: E501
    ...


def nanmedian(a: TreeT, axis: Tree = None, out: None = None, overwrite_input: bool = False, keepdims: bool = False) -> TreeT:  # noqa: E501
    ...


def nanmin(a: TreeT, axis: Tree = None, out: None = None, keepdims: bool = False, initial: Tree = None, where: Tree = None) -> TreeT:  # noqa: E501
    ...


def nanpercentile(a: TreeT, q: Tree, axis: Tree = None, out: None = None, overwrite_input: bool = False, method: str = 'linear', keepdims: bool = False, interpolation: None = None) -> TreeT:  # noqa: E501
    ...


def nanprod(a: TreeT, axis: Tree = None, dtype: Tree = None, out: None = None, keepdims: bool = False, initial: Tree = None, where: Tree = None) -> TreeT:  # noqa: E501
    ...


def nanquantile(a: TreeT, q: Tree, axis: Tree = None, out: None = None, overwrite_input: bool = False, method: str = 'linear', keepdims: bool = False, interpolation: None = None) -> TreeT:  # noqa: E501
    ...


def nanstd(a: TreeT, axis: Tree = None, dtype: Tree = None, out: None = None, ddof: int = 0, keepdims: bool = False, where: Tree = None) -> TreeT:  # noqa: E501
    ...


def nansum(a: TreeT, axis: Tree = None, dtype: Tree = None, out: None = None, keepdims: bool = False, initial: Tree = None, where: Tree = None) -> TreeT:  # noqa: E501
    ...


def nanvar(a: TreeT, axis: Tree = None, dtype: Tree = None, out: None = None, ddof: int = 0, keepdims: bool = False, where: Tree = None) -> TreeT:  # noqa: E501
    ...


def negative(x: TreeT, /) -> TreeT:  # noqa: E501
    ...


def nextafter(x1: TreeT, x2: TreeT, /) -> TreeT:  # noqa: E501
    ...


def nonzero(a: TreeT, *, size: int | None = None, fill_value: None | Tree | tuple[Tree, ...] = None) -> tuple[TreeT, ...]:  # noqa: E501
    ...


def not_equal(x1: TreeT, x2: TreeT, /) -> TreeT:  # noqa: E501
    ...


def ones(shape: Any, dtype: DTypeLike | None = None) -> Tree:  # noqa: E501
    ...


def ones_like(a: TreeT, dtype: DTypeLike | None = None, shape: Any = None) -> TreeT:  # noqa: E501
    ...


def outer(a: TreeT, b: Tree, out: None = None) -> TreeT:  # noqa: E501
    ...


def packbits(a: TreeT, axis: int | None = None, bitorder: str = 'big') -> TreeT:  # noqa: E501
    ...


def pad(array: TreeT, pad_width: PadValueLike[int | Tree | np.ndarray], mode: str | Callable[..., Any] = 'constant', **kwargs) -> TreeT:  # noqa: E501
    ...


def partition(a: TreeT, kth: int, axis: int = -1) -> TreeT:  # noqa: E501
    ...


def percentile(a: TreeT, q: Tree, axis: Tree = None, out: None = None, overwrite_input: bool = False, method: str = 'linear', keepdims: bool = False, interpolation: None = None) -> TreeT:  # noqa: E501
    ...


def piecewise(x: TreeT, condlist: Tree | Sequence[Tree], funclist: list[Tree | Callable[..., Tree]], *args, **kw) -> TreeT:  # noqa: E501
    ...


def place(arr: TreeT, mask: Tree, vals: Tree, *, inplace: bool = True) -> TreeT:  # noqa: E501
    ...


def poly(seq_of_zeros: TreeT) -> TreeT:  # noqa: E501
    ...


def polyadd(a1: TreeT, a2: Tree) -> TreeT:  # noqa: E501
    ...


def polyder(p: TreeT, m: int = 1) -> TreeT:  # noqa: E501
    ...


def polydiv(u: TreeT, v: Tree, *, trim_leading_zeros: bool = False) -> tuple[TreeT, TreeT]:  # noqa: E501
    ...


def polyfit(x: TreeT, y: Tree, deg: int, rcond: typing.Optional[float] = None, full: bool = False, w: typing.Optional[Tree] = None, cov: bool = False) -> TreeT:  # noqa: E501
    ...


def polyint(p: TreeT, m: int = 1, k: typing.Optional[int] = None) -> TreeT:  # noqa: E501
    ...


def polymul(a1: TreeT, a2: Tree, *, trim_leading_zeros: bool = False) -> TreeT:  # noqa: E501
    ...


def polysub(a1: TreeT, a2: Tree) -> TreeT:  # noqa: E501
    ...


def polyval(p: TreeT, x: Tree, *, unroll: int = 16) -> TreeT:  # noqa: E501
    ...


def positive(x: TreeT, /) -> TreeT:  # noqa: E501
    ...


def power(x1: TreeT, x2: TreeT, /) -> TreeT:  # noqa: E501
    ...


def prod(a: TreeT, axis: Tree = None, dtype: Tree = None, out: None = None, keepdims: bool = False, initial: Tree = None, where: Tree = None, promote_integers: bool = True) -> TreeT:  # noqa: E501
    ...


def ptp(a: TreeT, axis: Tree = None, out: None = None, keepdims: bool = False) -> TreeT:  # noqa: E501
    ...


def put(a: TreeT, ind: Tree, v: Tree, mode: str | None = None, *, inplace: bool = True) -> TreeT:  # noqa: E501
    ...


def quantile(a: TreeT, q: Tree, axis: Tree = None, out: None = None, overwrite_input: bool = False, method: str = 'linear', keepdims: bool = False, interpolation: None = None) -> TreeT:  # noqa: E501
    ...


def rad2deg(x: TreeT, /) -> TreeT:  # noqa: E501
    ...


def radians(x: TreeT, /) -> TreeT:  # noqa: E501
    ...


def ravel(a: TreeT, order: str = 'C') -> TreeT:  # noqa: E501
    ...


def ravel_multi_index(multi_index: Sequence[TreeT], dims: Sequence[int], mode: str = 'raise', order: str = 'C') -> TreeT:  # noqa: E501
    ...


def real(val: TreeT, /) -> TreeT:  # noqa: E501
    ...


def reciprocal(x: TreeT, /) -> TreeT:  # noqa: E501
    ...


def remainder(x1: TreeT, x2: TreeT, /) -> TreeT:  # noqa: E501
    ...


def repeat(a: TreeT, repeats: Tree, axis: int | None = None, *, total_repeat_length: int | None = None) -> TreeT:  # noqa: E501
    ...


def reshape(a: TreeT, newshape: DimSize | Shape, order: str = 'C') -> TreeT:  # noqa: E501
    ...


def resize(a: TreeT, new_shape: Shape) -> TreeT:  # noqa: E501
    ...


def reversed_broadcast(a: TreeT, to: Tree) -> TreeT:  # noqa: E501
    ...


def reversed_broadcasted_where_x(condition: Tree, x: TreeT, y: TreeT) -> TreeT:  # noqa: E501
    ...


def reversed_broadcasted_where_y(condition: Tree, x: TreeT, y: TreeT) -> TreeT:  # noqa: E501
    ...


def right_shift(x1: TreeT, x2: TreeT, /) -> TreeT:  # noqa: E501
    ...


def rint(x: TreeT, /) -> TreeT:  # noqa: E501
    ...


def roll(a: TreeT, shift: Tree | Sequence[int], axis: int | Sequence[int] | None = None) -> TreeT:  # noqa: E501
    ...


def rollaxis(a: TreeT, axis: int, start: int = 0) -> TreeT:  # noqa: E501
    ...


def roots(p: TreeT, *, strip_zeros: bool = True) -> TreeT:  # noqa: E501
    ...


def rot90(m: TreeT, k: int = 1, axes: tuple[int, int] = (0, 1)) -> TreeT:  # noqa: E501
    ...


def round(a: TreeT, decimals: int = 0, out: None = None) -> TreeT:  # noqa: E501
    ...


def round_(a: TreeT, decimals: int = 0, out: None = None) -> TreeT:  # noqa: E501
    ...


def searchsorted(a: TreeT, v: Tree, side: str = 'left', sorter: None = None, *, method: str = 'scan') -> TreeT:  # noqa: E501
    ...


def select(condlist: Sequence[TreeT], choicelist: Sequence[Tree], default: Tree = 0) -> TreeT:  # noqa: E501
    ...


def setdiff1d(ar1: TreeT, ar2: Tree, assume_unique: bool = False, *, size: typing.Optional[int] = None, fill_value: Tree = None) -> TreeT:  # noqa: E501
    ...


def setitem(a: TreeT, b: Tree, indices: None | int | slice | EllipsisType | Tree | tuple[None | int | slice | EllipsisType | Tree, ...]) -> TreeT:  # noqa: E501
    ...


def setxor1d(ar1: TreeT, ar2: Tree, assume_unique: bool = False) -> TreeT:  # noqa: E501
    ...


def shape(a: Tree) -> tuple[int, ...]:  # noqa: E501
    ...


def shape_dtype(a: TreeT) -> TreeT:  # noqa: E501
    ...


def sign(x: TreeT, /) -> TreeT:  # noqa: E501
    ...


def signbit(x: TreeT, /) -> TreeT:  # noqa: E501
    ...


def sin(x: TreeT, /) -> TreeT:  # noqa: E501
    ...


def sinc(x: TreeT, /) -> TreeT:  # noqa: E501
    ...


def single(x: Any) -> Tree:  # noqa: E501
    ...


def sinh(x: TreeT, /) -> TreeT:  # noqa: E501
    ...


def sort(a: TreeT, axis: int | None = -1, kind: str = 'quicksort', order: None = None) -> TreeT:  # noqa: E501
    ...


def sort_complex(a: TreeT) -> TreeT:  # noqa: E501
    ...


def split(ary: TreeT, indices_or_sections: int | Sequence[int] | Tree, axis: int = 0) -> list[TreeT]:  # noqa: E501
    ...


def sqrt(x: TreeT, /) -> TreeT:  # noqa: E501
    ...


def square(x: TreeT, /) -> TreeT:  # noqa: E501
    ...


def squeeze(a: TreeT, axis: int | Sequence[int] | None = None) -> TreeT:  # noqa: E501
    ...


def stack(*arrays: TreeT, axis: int = 0, out: None = None, dtype: DTypeLike | None = None) -> TreeT:  # noqa: E501
    ...


def std(a: TreeT, axis: Tree = None, dtype: Tree = None, out: None = None, ddof: int = 0, keepdims: bool = False, *, where: Tree = None) -> TreeT:  # noqa: E501
    ...


def subtract(x1: TreeT, x2: TreeT, /) -> TreeT:  # noqa: E501
    ...


def sum(a: TreeT, axis: Tree = None, dtype: Tree = None, out: None = None, keepdims: bool = False, initial: Tree = None, where: Tree = None, promote_integers: bool = True) -> TreeT:  # noqa: E501
    ...


def swapaxes(a: TreeT, axis1: int, axis2: int) -> TreeT:  # noqa: E501
    ...


def take(a: TreeT, indices: Tree, axis: int | None = None, out: None = None, mode: str | None = None, unique_indices: bool = False, indices_are_sorted: bool = False, fill_value: Tree | None = None) -> TreeT:  # noqa: E501
    ...


def take_along_axis(arr: TreeT, indices: Tree, axis: int | None, mode: str | lax.GatherScatterMode | None = None) -> TreeT:  # noqa: E501
    ...


def tan(x: TreeT, /) -> TreeT:  # noqa: E501
    ...


def tanh(x: TreeT, /) -> TreeT:  # noqa: E501
    ...


def tensordot(a: TreeT, b: Tree, axes: int | Sequence[int] | Sequence[Sequence[int]] = 2, *, precision: PrecisionLike = None, preferred_element_type: DTypeLike | None = None) -> TreeT:  # noqa: E501
    ...


def tile(A: TreeT, reps: DimSize | Sequence[DimSize]) -> TreeT:  # noqa: E501
    ...


def trace(a: TreeT, offset: int = 0, axis1: int = 0, axis2: int = 1, dtype: DTypeLike | None = None, out: None = None) -> TreeT:  # noqa: E501
    ...


def transpose(a: TreeT, axes: Sequence[int] | None = None) -> TreeT:  # noqa: E501
    ...


def tri(N: int, M: int | None = None, k: int = 0, dtype: Optional[DTypeLike] = None) -> Tree:  # noqa: E501
    ...


def tril(m: TreeT, k: int = 0) -> TreeT:  # noqa: E501
    ...


def tril_indices(n: int, k: int = 0, m: int | None = None) -> tuple[TreeT, TreeT]:  # noqa: E501
    ...


def tril_indices_from(arr: TreeT, k: int = 0) -> tuple[TreeT, TreeT]:  # noqa: E501
    ...


def triu(m: TreeT, k: int = 0) -> TreeT:  # noqa: E501
    ...


def triu_indices(n: int, k: int = 0, m: int | None = None) -> tuple[TreeT, TreeT]:  # noqa: E501
    ...


def triu_indices_from(arr: TreeT, k: int = 0) -> tuple[TreeT, TreeT]:  # noqa: E501
    ...


def true_divide(x1: TreeT, x2: TreeT, /) -> TreeT:  # noqa: E501
    ...


def trunc(x: TreeT) -> TreeT:  # noqa: E501
    ...


def uint(x: Any) -> Tree:  # noqa: E501
    ...


def uint16(x: Any) -> Tree:  # noqa: E501
    ...


def uint32(x: Any) -> Tree:  # noqa: E501
    ...


def uint4(x: Any) -> Tree:  # noqa: E501
    ...


def uint64(x: Any) -> Tree:  # noqa: E501
    ...


def uint8(x: Any) -> Tree:  # noqa: E501
    ...


def union1d(ar1: TreeT, ar2: Tree, *, size: typing.Optional[int] = None, fill_value: Tree = None) -> TreeT:  # noqa: E501
    ...


def unique(ar: Tree, return_index: bool = False, return_inverse: bool = False, return_counts: bool = False, axis: typing.Optional[int] = None, *, size: typing.Optional[int] = None, fill_value: Tree = None):  # noqa: E501
    ...


def unpackbits(a: TreeT, axis: int | None = None, count: int | None = None, bitorder: str = 'big') -> TreeT:  # noqa: E501
    ...


def unravel_index(indices: TreeT, shape: Shape) -> tuple[TreeT, ...]:  # noqa: E501
    ...


def unwrap(p: TreeT, discont: Tree | None = None, axis: int = -1, period: Tree = 6.283185307179586) -> TreeT:  # noqa: E501
    ...


def vander(x: TreeT, N: int | None = None, increasing: bool = False) -> TreeT:  # noqa: E501
    ...


def var(a: TreeT, axis: Tree = None, dtype: Tree = None, out: None = None, ddof: int = 0, keepdims: bool = False, *, where: Tree = None) -> TreeT:  # noqa: E501
    ...


def vdot(a: TreeT, b: Tree, *, precision: PrecisionLike = None, preferred_element_type: DTypeLike | None = None) -> TreeT:  # noqa: E501
    ...


def vsplit(ary: TreeT, indices_or_sections: int | Sequence[int] | Tree) -> list[TreeT]:  # noqa: E501
    ...


def vstack(*tup: TreeT, dtype: DTypeLike | None = None) -> TreeT:  # noqa: E501
    ...


def where(condition: Tree, x: TreeT | None = None, y: TreeT | None = None, *, size: int | None = None, fill_value: None | TreeT | tuple[TreeT, ...] = None) -> TreeT | tuple[TreeT, ...]:  # noqa: E501
    ...


def zeros(shape: Any, dtype: DTypeLike | None = None) -> Tree:  # noqa: E501
    ...


def zeros_like(a: TreeT, dtype: DTypeLike | None = None, shape: Any = None) -> TreeT:  # noqa: E501
    ...


__all__ = [
    'abs',
    'absolute',
    'add',
    'all',
    'allclose',
    'amax',
    'amin',
    'angle',
    'any',
    'append',
    'apply_along_axis',
    'apply_over_axes',
    'arange',
    'arccos',
    'arccosh',
    'arcsin',
    'arcsinh',
    'arctan',
    'arctan2',
    'arctanh',
    'argmax',
    'argmin',
    'argpartition',
    'argsort',
    'argwhere',
    'around',
    'array',
    'array_equal',
    'array_equiv',
    'array_split',
    'asarray',
    'asis',
    'atleast_1d',
    'atleast_2d',
    'atleast_3d',
    'average',
    'bartlett',
    'bfloat16',
    'bincount',
    'bitwise_and',
    'bitwise_count',
    'bitwise_not',
    'bitwise_or',
    'bitwise_xor',
    'blackman',
    'block',
    'bool_',
    'broadcast_arrays',
    'broadcast_to',
    'cbrt',
    'cdouble',
    'ceil',
    'choose',
    'clip',
    'column_stack',
    'complex128',
    'complex64',
    'complex_',
    'compress',
    'concatenate',
    'conj',
    'conjugate',
    'convolve',
    'copy',
    'copysign',
    'corrcoef',
    'correlate',
    'cos',
    'cosh',
    'count_nonzero',
    'cov',
    'csingle',
    'cumprod',
    'cumsum',
    'deg2rad',
    'degrees',
    'delete',
    'diag',
    'diag_indices',
    'diag_indices_from',
    'diagflat',
    'diagonal',
    'diff',
    'digitize',
    'distance',
    'divide',
    'divmod',
    'dot',
    'double',
    'dsplit',
    'dstack',
    'duplicate',
    'ediff1d',
    'einsum',
    'empty',
    'empty_like',
    'equal',
    'exp',
    'exp2',
    'expand_dims',
    'expm1',
    'extract',
    'eye',
    'fabs',
    'fill_diagonal',
    'fix',
    'flatnonzero',
    'flip',
    'fliplr',
    'flipud',
    'float16',
    'float32',
    'float64',
    'float8_e4m3b11fnuz',
    'float8_e4m3fn',
    'float8_e4m3fnuz',
    'float8_e5m2',
    'float8_e5m2fnuz',
    'float_',
    'float_power',
    'floor',
    'floor_divide',
    'fmax',
    'fmin',
    'fmod',
    'frexp',
    'from_dlpack',
    'frombuffer',
    'fromfunction',
    'fromstring',
    'full',
    'full_like',
    'gcd',
    'geomspace',
    'getitem',
    'gradient',
    'greater',
    'greater_equal',
    'hamming',
    'hanning',
    'heaviside',
    'histogram',
    'histogram2d',
    'histogram_bin_edges',
    'histogramdd',
    'hsplit',
    'hstack',
    'hypot',
    'i0',
    'identity',
    'imag',
    'indices',
    'inner',
    'insert',
    'int16',
    'int32',
    'int4',
    'int64',
    'int8',
    'int_',
    'interp',
    'intersect1d',
    'invert',
    'isclose',
    'iscomplex',
    'isfinite',
    'isin',
    'isinf',
    'isnan',
    'isneginf',
    'isposinf',
    'isreal',
    'ix_',
    'kaiser',
    'kron',
    'lcm',
    'ldexp',
    'left_shift',
    'less',
    'less_equal',
    'lexsort',
    'linspace',
    'load',
    'log',
    'log10',
    'log1p',
    'log2',
    'logaddexp',
    'logaddexp2',
    'logical_and',
    'logical_not',
    'logical_or',
    'logical_xor',
    'logspace',
    'matmul',
    'matrix_transpose',
    'max',
    'maximum',
    'mean',
    'median',
    'meshgrid',
    'min',
    'minimum',
    'mod',
    'modf',
    'moveaxis',
    'multiply',
    'nan_to_num',
    'nanargmax',
    'nanargmin',
    'nancumprod',
    'nancumsum',
    'nanmax',
    'nanmean',
    'nanmedian',
    'nanmin',
    'nanpercentile',
    'nanprod',
    'nanquantile',
    'nanstd',
    'nansum',
    'nanvar',
    'negative',
    'nextafter',
    'nonzero',
    'not_equal',
    'ones',
    'ones_like',
    'outer',
    'packbits',
    'pad',
    'partition',
    'percentile',
    'piecewise',
    'place',
    'poly',
    'polyadd',
    'polyder',
    'polydiv',
    'polyfit',
    'polyint',
    'polymul',
    'polysub',
    'polyval',
    'positive',
    'power',
    'prod',
    'ptp',
    'put',
    'quantile',
    'rad2deg',
    'radians',
    'ravel',
    'ravel_multi_index',
    'real',
    'reciprocal',
    'remainder',
    'repeat',
    'reshape',
    'resize',
    'reversed_broadcast',
    'reversed_broadcasted_where_x',
    'reversed_broadcasted_where_y',
    'right_shift',
    'rint',
    'roll',
    'rollaxis',
    'roots',
    'rot90',
    'round',
    'round_',
    'searchsorted',
    'select',
    'setdiff1d',
    'setitem',
    'setxor1d',
    'shape',
    'shape_dtype',
    'sign',
    'signbit',
    'sin',
    'sinc',
    'single',
    'sinh',
    'sort',
    'sort_complex',
    'split',
    'sqrt',
    'square',
    'squeeze',
    'stack',
    'std',
    'subtract',
    'sum',
    'swapaxes',
    'take',
    'take_along_axis',
    'tan',
    'tanh',
    'tensordot',
    'tile',
    'trace',
    'transpose',
    'tri',
    'tril',
    'tril_indices',
    'tril_indices_from',
    'triu',
    'triu_indices',
    'triu_indices_from',
    'true_divide',
    'trunc',
    'uint',
    'uint16',
    'uint32',
    'uint4',
    'uint64',
    'uint8',
    'union1d',
    'unique',
    'unpackbits',
    'unravel_index',
    'unwrap',
    'vander',
    'var',
    'vdot',
    'vsplit',
    'vstack',
    'where',
    'zeros',
    'zeros_like',
]
