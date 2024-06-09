from ._lax_head import Any, Callable, DTypeLike, DotDimensionNumbers, GatherDimensionNumbers, Optional, PrecisionLike, RoundingMethod, ScatterDimensionNumbers, Sequence, Shape, Tree, TreeT, Union, collections, jax, np, typing  # noqa: E501


def abs(x: TreeT) -> TreeT:  # noqa: E501
    ...


def acos(x: TreeT) -> TreeT:  # noqa: E501
    ...


def acosh(x: TreeT) -> TreeT:  # noqa: E501
    ...


def add(x: TreeT, y: Tree) -> TreeT:  # noqa: E501
    ...


def approx_max_k(operand: TreeT, k: int, reduction_dimension: int = -1, recall_target: float = 0.95, reduction_input_size_override: int = -1, aggregate_to_topk: bool = True) -> tuple[TreeT, TreeT]:  # noqa: E501
    ...


def approx_min_k(operand: TreeT, k: int, reduction_dimension: int = -1, recall_target: float = 0.95, reduction_input_size_override: int = -1, aggregate_to_topk: bool = True) -> tuple[TreeT, TreeT]:  # noqa: E501
    ...


def argmax(operand: TreeT, axis: int, index_dtype: DTypeLike) -> TreeT:  # noqa: E501
    ...


def argmin(operand: TreeT, axis: int, index_dtype: DTypeLike) -> TreeT:  # noqa: E501
    ...


def asin(x: TreeT) -> TreeT:  # noqa: E501
    ...


def asinh(x: TreeT) -> TreeT:  # noqa: E501
    ...


def atan(x: TreeT) -> TreeT:  # noqa: E501
    ...


def atan2(x: TreeT, y: Tree) -> TreeT:  # noqa: E501
    ...


def atanh(x: TreeT) -> TreeT:  # noqa: E501
    ...


def batch_matmul(lhs: TreeT, rhs: Tree, precision: PrecisionLike = None) -> TreeT:  # noqa: E501
    ...


def bessel_i0e(x: TreeT) -> TreeT:  # noqa: E501
    ...


def bessel_i1e(x: TreeT) -> TreeT:  # noqa: E501
    ...


def betainc(a: TreeT, b: Tree, x: Tree) -> TreeT:  # noqa: E501
    ...


def bitcast_convert_type(operand: TreeT, new_dtype: DTypeLike) -> TreeT:  # noqa: E501
    ...


def bitwise_and(x: TreeT, y: Tree) -> TreeT:  # noqa: E501
    ...


def bitwise_not(x: TreeT) -> TreeT:  # noqa: E501
    ...


def bitwise_or(x: TreeT, y: Tree) -> TreeT:  # noqa: E501
    ...


def bitwise_xor(x: TreeT, y: Tree) -> TreeT:  # noqa: E501
    ...


def broadcast(operand: TreeT, sizes: Sequence[int]) -> TreeT:  # noqa: E501
    ...


def broadcast_in_dim(operand: TreeT, shape: Shape, broadcast_dimensions: Sequence[int]) -> TreeT:  # noqa: E501
    ...


def broadcast_to_rank(x: TreeT, rank: int) -> TreeT:  # noqa: E501
    ...


def broadcasted_iota(dtype: DTypeLike, shape: Shape, dimension: int) -> Tree:  # noqa: E501
    ...


def cbrt(x: TreeT) -> TreeT:  # noqa: E501
    ...


def ceil(x: TreeT) -> TreeT:  # noqa: E501
    ...


def clamp(min: TreeT, x: Tree, max: Tree) -> TreeT:  # noqa: E501
    ...


def clz(x: TreeT) -> TreeT:  # noqa: E501
    ...


def collapse(operand: TreeT, start_dimension: int, stop_dimension: Optional[int] = None) -> TreeT:  # noqa: E501
    ...


def complex(x: TreeT, y: Tree) -> TreeT:  # noqa: E501
    ...


def concatenate(*arrays: TreeT, axis: int | None = 0, dtype: DTypeLike | None = None) -> TreeT:  # noqa: E501
    ...


def conj(x: TreeT) -> TreeT:  # noqa: E501
    ...


def conv(lhs: TreeT, rhs: Tree, window_strides: collections.abc.Sequence[int], padding: str, precision: Tree = None, preferred_element_type: Tree = None) -> TreeT:  # noqa: E501
    ...


def conv_general_dilated(lhs: TreeT, rhs: Tree, window_strides: collections.abc.Sequence[int], padding: Tree, lhs_dilation: typing.Optional[collections.abc.Sequence[int]] = None, rhs_dilation: typing.Optional[collections.abc.Sequence[int]] = None, dimension_numbers: Tree = None, feature_group_count: int = 1, batch_group_count: int = 1, precision: Tree = None, preferred_element_type: Tree = None) -> TreeT:  # noqa: E501
    ...


def conv_general_dilated_local(lhs: TreeT, rhs: Tree, window_strides: collections.abc.Sequence[int], padding: Tree, filter_shape: collections.abc.Sequence[int], lhs_dilation: typing.Optional[collections.abc.Sequence[int]] = None, rhs_dilation: typing.Optional[collections.abc.Sequence[int]] = None, dimension_numbers: Tree = None, precision: Tree = None) -> TreeT:  # noqa: E501
    ...


def conv_general_dilated_patches(lhs: TreeT, filter_shape: collections.abc.Sequence[int], window_strides: collections.abc.Sequence[int], padding: Tree, lhs_dilation: typing.Optional[collections.abc.Sequence[int]] = None, rhs_dilation: typing.Optional[collections.abc.Sequence[int]] = None, dimension_numbers: Tree = None, precision: typing.Optional[jax._src.lax.lax.Precision] = None, preferred_element_type: typing.Optional[typing.Any] = None) -> TreeT:  # noqa: E501
    ...


def conv_transpose(lhs: TreeT, rhs: Tree, strides: collections.abc.Sequence[int], padding: Tree, rhs_dilation: typing.Optional[collections.abc.Sequence[int]] = None, dimension_numbers: Tree = None, transpose_kernel: bool = False, precision: Tree = None, preferred_element_type: Tree = None) -> TreeT:  # noqa: E501
    ...


def conv_with_general_padding(lhs: TreeT, rhs: Tree, window_strides: collections.abc.Sequence[int], padding: Tree, lhs_dilation: typing.Optional[collections.abc.Sequence[int]], rhs_dilation: typing.Optional[collections.abc.Sequence[int]], precision: Tree = None, preferred_element_type: Tree = None) -> TreeT:  # noqa: E501
    ...


def convert_element_type(operand: TreeT, new_dtype: DTypeLike) -> TreeT:  # noqa: E501
    ...


def cos(x: TreeT) -> TreeT:  # noqa: E501
    ...


def cosh(x: TreeT) -> TreeT:  # noqa: E501
    ...


def cumlogsumexp(operand: TreeT, axis: int = 0, reverse: bool = False) -> TreeT:  # noqa: E501
    ...


def cummax(operand: TreeT, axis: int = 0, reverse: bool = False) -> TreeT:  # noqa: E501
    ...


def cummin(operand: TreeT, axis: int = 0, reverse: bool = False) -> TreeT:  # noqa: E501
    ...


def cumprod(operand: TreeT, axis: int = 0, reverse: bool = False) -> TreeT:  # noqa: E501
    ...


def cumsum(operand: TreeT, axis: int = 0, reverse: bool = False) -> TreeT:  # noqa: E501
    ...


def digamma(x: TreeT) -> TreeT:  # noqa: E501
    ...


def div(x: TreeT, y: Tree) -> TreeT:  # noqa: E501
    ...


def dot(lhs: TreeT, rhs: Tree, precision: PrecisionLike = None, preferred_element_type: Optional[DTypeLike] = None) -> TreeT:  # noqa: E501
    ...


def dot_general(lhs: TreeT, rhs: Tree, dimension_numbers: DotDimensionNumbers, precision: PrecisionLike = None, preferred_element_type: Optional[DTypeLike] = None) -> TreeT:  # noqa: E501
    ...


def dynamic_index_in_dim(operand: TreeT, index: Tree, axis: int = 0, keepdims: bool = True) -> TreeT:  # noqa: E501
    ...


def dynamic_slice(operand: TreeT, start_indices: Tree, slice_sizes: collections.abc.Sequence[typing.Union[int, typing.Any]]) -> TreeT:  # noqa: E501
    ...


def dynamic_slice_in_dim(operand: TreeT, start_index: Tree, slice_size: int, axis: int = 0) -> TreeT:  # noqa: E501
    ...


def dynamic_update_index_in_dim(operand: TreeT, update: Tree, index: Tree, axis: int) -> TreeT:  # noqa: E501
    ...


def dynamic_update_slice(operand: TreeT, update: Tree, start_indices: Tree) -> TreeT:  # noqa: E501
    ...


def dynamic_update_slice_in_dim(operand: TreeT, update: Tree, start_index: Tree, axis: int) -> TreeT:  # noqa: E501
    ...


def eq(x: TreeT, y: Tree) -> TreeT:  # noqa: E501
    ...


def erf(x: TreeT) -> TreeT:  # noqa: E501
    ...


def erf_inv(x: TreeT) -> TreeT:  # noqa: E501
    ...


def erfc(x: TreeT) -> TreeT:  # noqa: E501
    ...


def exp(x: TreeT) -> TreeT:  # noqa: E501
    ...


def exp2(x: TreeT) -> TreeT:  # noqa: E501
    ...


def expand_dims(array: TreeT, dimensions: Sequence[int]) -> TreeT:  # noqa: E501
    ...


def expm1(x: TreeT) -> TreeT:  # noqa: E501
    ...


def fft(x, fft_type: Tree, fft_lengths: collections.abc.Sequence[int]):  # noqa: E501
    ...


def floor(x: TreeT) -> TreeT:  # noqa: E501
    ...


def full(shape: Shape, fill_value: TreeT, dtype: Optional[DTypeLike] = None) -> TreeT:  # noqa: E501
    ...


def full_like(x: Union[TreeT, TreeT], fill_value: Tree, dtype: Optional[DTypeLike] = None, shape: Optional[Shape] = None) -> TreeT:  # noqa: E501
    ...


def gather(operand: TreeT, start_indices: Tree, dimension_numbers: GatherDimensionNumbers, slice_sizes: collections.abc.Sequence[typing.Union[int, typing.Any]], *, unique_indices: bool = False, indices_are_sorted: bool = False, mode: Tree = None, fill_value=None) -> TreeT:  # noqa: E501
    ...


def ge(x: TreeT, y: Tree) -> TreeT:  # noqa: E501
    ...


def gt(x: TreeT, y: Tree) -> TreeT:  # noqa: E501
    ...


def igamma(a: TreeT, x: Tree) -> TreeT:  # noqa: E501
    ...


def igamma_grad_a(a: TreeT, x: Tree) -> TreeT:  # noqa: E501
    ...


def igammac(a: TreeT, x: Tree) -> TreeT:  # noqa: E501
    ...


def imag(x: TreeT) -> TreeT:  # noqa: E501
    ...


def index_in_dim(operand: TreeT, index: int, axis: int = 0, keepdims: bool = True) -> TreeT:  # noqa: E501
    ...


def index_take(src: TreeT, idxs: Tree, axes: collections.abc.Sequence[int]) -> TreeT:  # noqa: E501
    ...


def integer_pow(x: TreeT, y: int) -> TreeT:  # noqa: E501
    ...


def iota(dtype: DTypeLike, size: int) -> Tree:  # noqa: E501
    ...


def is_finite(x: TreeT) -> TreeT:  # noqa: E501
    ...


def le(x: TreeT, y: Tree) -> TreeT:  # noqa: E501
    ...


def lgamma(x: TreeT) -> TreeT:  # noqa: E501
    ...


def log(x: TreeT) -> TreeT:  # noqa: E501
    ...


def log1p(x: TreeT) -> TreeT:  # noqa: E501
    ...


def logistic(x: TreeT) -> TreeT:  # noqa: E501
    ...


def lt(x: TreeT, y: Tree) -> TreeT:  # noqa: E501
    ...


def max(x: TreeT, y: Tree) -> TreeT:  # noqa: E501
    ...


def min(x: TreeT, y: Tree) -> TreeT:  # noqa: E501
    ...


def mul(x: TreeT, y: Tree) -> TreeT:  # noqa: E501
    ...


def ne(x: TreeT, y: Tree) -> TreeT:  # noqa: E501
    ...


def neg(x: TreeT) -> TreeT:  # noqa: E501
    ...


def nextafter(x1: TreeT, x2: Tree) -> TreeT:  # noqa: E501
    ...


def pad(operand: TreeT, padding_value: Tree, padding_config: Sequence[tuple[int, int, int]]) -> TreeT:  # noqa: E501
    ...


def polygamma(m: TreeT, x: Tree) -> TreeT:  # noqa: E501
    ...


def population_count(x: TreeT) -> TreeT:  # noqa: E501
    ...


def pow(x: TreeT, y: Tree) -> TreeT:  # noqa: E501
    ...


def random_gamma_grad(a: TreeT, x: Tree) -> TreeT:  # noqa: E501
    ...


def real(x: TreeT) -> TreeT:  # noqa: E501
    ...


def reciprocal(x: TreeT) -> TreeT:  # noqa: E501
    ...


def reduce_precision(operand: Union[float, TreeT], exponent_bits: int, mantissa_bits: int) -> TreeT:  # noqa: E501
    ...


def reduce_window(operand, init_value, computation: Callable, window_dimensions: collections.abc.Sequence[typing.Union[int, typing.Any]], window_strides: collections.abc.Sequence[int], padding: TreeT, base_dilation: typing.Optional[collections.abc.Sequence[int]] = None, window_dilation: typing.Optional[collections.abc.Sequence[int]] = None) -> TreeT:  # noqa: E501
    ...


def rem(x: TreeT, y: Tree) -> TreeT:  # noqa: E501
    ...


def reshape(operand: TreeT, new_sizes: Shape, dimensions: Optional[Sequence[int]] = None) -> TreeT:  # noqa: E501
    ...


def rev(operand: TreeT, dimensions: Sequence[int]) -> TreeT:  # noqa: E501
    ...


def round(x: TreeT, rounding_method: RoundingMethod = RoundingMethod.AWAY_FROM_ZERO) -> TreeT:  # noqa: E501
    ...


def rsqrt(x: TreeT) -> TreeT:  # noqa: E501
    ...


def scatter(operand: TreeT, scatter_indices: Tree, updates: Tree, dimension_numbers: ScatterDimensionNumbers, *, indices_are_sorted: bool = False, unique_indices: bool = False, mode: Tree = None) -> TreeT:  # noqa: E501
    ...


def scatter_add(operand: TreeT, scatter_indices: Tree, updates: Tree, dimension_numbers: ScatterDimensionNumbers, *, indices_are_sorted: bool = False, unique_indices: bool = False, mode: Tree = None) -> TreeT:  # noqa: E501
    ...


def scatter_apply(operand: TreeT, scatter_indices: Tree, func: Callable, dimension_numbers: ScatterDimensionNumbers, *, update_shape: collections.abc.Sequence[typing.Union[int, typing.Any]] = (), indices_are_sorted: bool = False, unique_indices: bool = False, mode: Tree = None) -> TreeT:  # noqa: E501
    ...


def scatter_max(operand: TreeT, scatter_indices: Tree, updates: Tree, dimension_numbers: ScatterDimensionNumbers, *, indices_are_sorted: bool = False, unique_indices: bool = False, mode: Tree = None) -> TreeT:  # noqa: E501
    ...


def scatter_min(operand: TreeT, scatter_indices: Tree, updates: Tree, dimension_numbers: ScatterDimensionNumbers, *, indices_are_sorted: bool = False, unique_indices: bool = False, mode: Tree = None) -> TreeT:  # noqa: E501
    ...


def scatter_mul(operand: TreeT, scatter_indices: Tree, updates: Tree, dimension_numbers: ScatterDimensionNumbers, *, indices_are_sorted: bool = False, unique_indices: bool = False, mode: Tree = None) -> TreeT:  # noqa: E501
    ...


def select(pred: TreeT, on_true: Tree, on_false: Tree) -> TreeT:  # noqa: E501
    ...


def select_n(which: TreeT, *cases: Tree) -> TreeT:  # noqa: E501
    ...


def shift_left(x: TreeT, y: Tree) -> TreeT:  # noqa: E501
    ...


def shift_right_arithmetic(x: TreeT, y: Tree) -> TreeT:  # noqa: E501
    ...


def shift_right_logical(x: TreeT, y: Tree) -> TreeT:  # noqa: E501
    ...


def sign(x: TreeT) -> TreeT:  # noqa: E501
    ...


def sin(x: TreeT) -> TreeT:  # noqa: E501
    ...


def sinh(x: TreeT) -> TreeT:  # noqa: E501
    ...


def slice(operand: TreeT, start_indices: collections.abc.Sequence[int], limit_indices: collections.abc.Sequence[int], strides: typing.Optional[collections.abc.Sequence[int]] = None) -> TreeT:  # noqa: E501
    ...


def slice_in_dim(operand: TreeT, start_index: typing.Optional[int], limit_index: typing.Optional[int], stride: int = 1, axis: int = 0) -> TreeT:  # noqa: E501
    ...


def sort(operand: Union[TreeT, Sequence[TreeT]], dimension: int = -1, is_stable: bool = True, num_keys: int = 1) -> Union[TreeT, tuple[TreeT, ...]]:  # noqa: E501
    ...


def sort_key_val(keys: TreeT, values: Tree, dimension: int = -1, is_stable: bool = True) -> tuple[TreeT, TreeT]:  # noqa: E501
    ...


def sqrt(x: TreeT) -> TreeT:  # noqa: E501
    ...


def square(x: TreeT) -> TreeT:  # noqa: E501
    ...


def squeeze(array: TreeT, dimensions: Sequence[int]) -> TreeT:  # noqa: E501
    ...


def stop_gradient(x: TreeT) -> TreeT:  # noqa: E501
    ...


def sub(x: TreeT, y: Tree) -> TreeT:  # noqa: E501
    ...


def tan(x: TreeT) -> TreeT:  # noqa: E501
    ...


def tanh(x: TreeT) -> TreeT:  # noqa: E501
    ...


def tie_in(x: Any, y: TreeT) -> TreeT:  # noqa: E501
    ...


def top_k(operand: TreeT, k: int) -> tuple[TreeT, TreeT]:  # noqa: E501
    ...


def transpose(operand: TreeT, permutation: Sequence[int] | np.ndarray) -> TreeT:  # noqa: E501
    ...


def zeros_like_array(x: TreeT) -> TreeT:  # noqa: E501
    ...


def zeta(x: TreeT, q: Tree) -> TreeT:  # noqa: E501
    ...


__all__ = [
    'abs',
    'acos',
    'acosh',
    'add',
    'approx_max_k',
    'approx_min_k',
    'argmax',
    'argmin',
    'asin',
    'asinh',
    'atan',
    'atan2',
    'atanh',
    'batch_matmul',
    'bessel_i0e',
    'bessel_i1e',
    'betainc',
    'bitcast_convert_type',
    'bitwise_and',
    'bitwise_not',
    'bitwise_or',
    'bitwise_xor',
    'broadcast',
    'broadcast_in_dim',
    'broadcast_to_rank',
    'broadcasted_iota',
    'cbrt',
    'ceil',
    'clamp',
    'clz',
    'collapse',
    'complex',
    'concatenate',
    'conj',
    'conv',
    'conv_general_dilated',
    'conv_general_dilated_local',
    'conv_general_dilated_patches',
    'conv_transpose',
    'conv_with_general_padding',
    'convert_element_type',
    'cos',
    'cosh',
    'cumlogsumexp',
    'cummax',
    'cummin',
    'cumprod',
    'cumsum',
    'digamma',
    'div',
    'dot',
    'dot_general',
    'dynamic_index_in_dim',
    'dynamic_slice',
    'dynamic_slice_in_dim',
    'dynamic_update_index_in_dim',
    'dynamic_update_slice',
    'dynamic_update_slice_in_dim',
    'eq',
    'erf',
    'erf_inv',
    'erfc',
    'exp',
    'exp2',
    'expand_dims',
    'expm1',
    'fft',
    'floor',
    'full',
    'full_like',
    'gather',
    'ge',
    'gt',
    'igamma',
    'igamma_grad_a',
    'igammac',
    'imag',
    'index_in_dim',
    'index_take',
    'integer_pow',
    'iota',
    'is_finite',
    'le',
    'lgamma',
    'log',
    'log1p',
    'logistic',
    'lt',
    'max',
    'min',
    'mul',
    'ne',
    'neg',
    'nextafter',
    'pad',
    'polygamma',
    'population_count',
    'pow',
    'random_gamma_grad',
    'real',
    'reciprocal',
    'reduce_precision',
    'reduce_window',
    'rem',
    'reshape',
    'rev',
    'round',
    'rsqrt',
    'scatter',
    'scatter_add',
    'scatter_apply',
    'scatter_max',
    'scatter_min',
    'scatter_mul',
    'select',
    'select_n',
    'shift_left',
    'shift_right_arithmetic',
    'shift_right_logical',
    'sign',
    'sin',
    'sinh',
    'slice',
    'slice_in_dim',
    'sort',
    'sort_key_val',
    'sqrt',
    'square',
    'squeeze',
    'stop_gradient',
    'sub',
    'tan',
    'tanh',
    'tie_in',
    'top_k',
    'transpose',
    'zeros_like_array',
    'zeta',
]
