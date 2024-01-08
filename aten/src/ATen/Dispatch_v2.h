#include <ATen/Dispatch.h>

// This is a new implementation of the AT_DISPATCH macro family from
// ATen/Dispatch.h
//
// The intended usage is:
//
//  ScalarType scalar_type;
//
//  AT_DISPATCH_V2(
//    scalar_type,
//    "debug string",
//    AT_WRAP([&] {
//      ... code to specialize with scalar_t ...
//    }),
//    kHalf,
//    AT_EXPAND(AT_ALL_TYPES),
//    ... as many types arguments as needed ...
//  )
//
// For example, given an old style:
//
//  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND2(
//    kComplexHalf,
//    kHalf,
//    self.scalar_type(),
//    "_local_scalar_dense_cpu",
//    [&] {
//      scalar_t value = *self.data_ptr<scalar_t>();
//      r = Scalar(value);
//    }
//  )
//
// You now write:
//
//  AT_DISPATCH_V2(
//    self.scalar_type(),
//    "_local_scalar_dense_cpu",
//    AT_WRAP([&] {
//      scalar_t value = *self.data_ptr<scalar_t>();
//      r = Scalar(value);
//    }),
//    AT_EXPAND(AT_ALL_TYPES),
//    AT_EXPAND(AT_COMPLEX_TYPES),
//    kComplexHalf,
//    kHalf,
//  )
//
// Notably, it sports the following improvements:
//
//  - It is not necessary to specify the arity (e.g.,
//    AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND{2,3,4,...})
//    when using the macro
//
//  - It is not necessary to specify each dtype individually; if
//    there is a set of related dtypes and you want to dispatch
//    over all of them, you can simply say, e.g., AT_EXPAND(AT_INTEGRAL_TYPES)
//    in your argument list.
//
// However, you must remember to wrap the payload body in AT_WRAP, or commas
// inside your lambda will be improperly handled.  Furthermore, if you more
// entries to ScalarType than can be supported by this macro, it will fail
// with an obscure error (due to attempting to concatenate AT_AP with
// something that is not a number).
//
// The implementation strategy is to use the count arguments trick
// (e.g., as described in https://stackoverflow.com/a/2124385/23845)
// to discover how many dtypes have been passed, and then dispatch to a
// hand-written macro for each arity that applies as many DISPATCH_CASE as
// necessary.  The hand-written macros can be regenerated for other arities
// with the script below.
//
// There is some delicacy in the implementation in controlling when
// macro expansion occurs, mediated with AT_EXPAND and AT_GUARD.  I mostly
// relied on GPT4 to help me get it right.

// Public API macros

// See documentation above
#define AT_DISPATCH_V2(TYPE, NAME, BODY, ...) \
  AT_DISPATCH_SWITCH(TYPE, NAME, AT_AP_VAR(AT_WRAP(BODY), TYPE, __VA_ARGS__))

// This macro lets you pass an arbitrary expression that may contain internal
// commas to another macro without having the commas causing the expression
// to be interpreted as being multiple arguments
#define AT_WRAP(...) __VA_ARGS__

#define AT_FLOAT8_TYPES \
  kFloat8_e5m2, kFloat8_e5m2fnuz, kFloat8_e4m3fn, kFloat8_e4m3fnuz

#define AT_INTEGRAL_TYPES kByte, kChar, kInt, kLong, kShort
#define AT_FLOATING_TYPES kDouble, kFloat
#define AT_BAREBONES_UNSIGNED_TYPES kUInt16, kUInt32, kUInt64
#define AT_COMPLEX_TYPES kComplexDouble, kComplexFloat
#define AT_QINT_TYPES kQInt8, kQUInt8, kQInt32
// NB: not *actually* all types
#define AT_ALL_TYPES AT_EXPAND(AT_INTEGRAL_TYPES), AT_EXPAND(AT_FLOATING_TYPES)
#define AT_ALL_TYPES_AND_COMPLEX \
  AT_EXPAND(AT_ALL_TYPES), AT_EXPAND(AT_COMPLEX_TYPES)

// Helper macros

#define AT_AP_VAR(N, T, ...) \
  AT_EXPAND(AT_CONCAT(AT_AP, AT_NUM_ARGS(__VA_ARGS__))(AT_WRAP(N), __VA_ARGS__))
#define AT_CONCAT(a, b) AT_CONCAT_AUX(a, b)
#define AT_CONCAT_AUX(a, b) a##b
#define AT_EXPAND(X) X

// Ensure we never have too many scalar types for the expansion here to
// support.  To bump this, you must regenerate the macros below.
static_assert(static_cast<int>(c10::ScalarType::NumOptions) < 32);

// Python code to regenerate generate code below:
#if 0

num_args = 32

nums = ', '.join(str(i) for i in reversed(range(num_args+1)))
args = ', '.join(f'_{i}' for i in range(1, num_args+1))

print(f'#define AT_NUM_ARGS(...) AT_EXPAND(AT_NUM_ARGS_AUX(__VA_ARGS__, {nums}))')
print(f'#define AT_NUM_ARGS_AUX({args}, N, ...) N')

for i in range(1, num_args+1):
    args = ', '.join(f'_{i}' for i in range(1, i+1))
    cases = ' '.join([f'AT_DISPATCH_CASE(_{j}, N)' for j in range(1, i+1)])
    print(f'#define AT_AP{i}(N, {args}) {cases}')

#endif

// Begin generated code
// clang-format off

#define AT_NUM_ARGS(...) AT_EXPAND(AT_NUM_ARGS_AUX(__VA_ARGS__, 32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0))
#define AT_NUM_ARGS_AUX(_1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16, _17, _18, _19, _20, _21, _22, _23, _24, _25, _26, _27, _28, _29, _30, _31, _32, N, ...) N
#define AT_AP1(N, _1) AT_DISPATCH_CASE(_1, N)
#define AT_AP2(N, _1, _2) AT_DISPATCH_CASE(_1, N) AT_DISPATCH_CASE(_2, N)
#define AT_AP3(N, _1, _2, _3) AT_DISPATCH_CASE(_1, N) AT_DISPATCH_CASE(_2, N) AT_DISPATCH_CASE(_3, N)
#define AT_AP4(N, _1, _2, _3, _4) AT_DISPATCH_CASE(_1, N) AT_DISPATCH_CASE(_2, N) AT_DISPATCH_CASE(_3, N) AT_DISPATCH_CASE(_4, N)
#define AT_AP5(N, _1, _2, _3, _4, _5) AT_DISPATCH_CASE(_1, N) AT_DISPATCH_CASE(_2, N) AT_DISPATCH_CASE(_3, N) AT_DISPATCH_CASE(_4, N) AT_DISPATCH_CASE(_5, N)
#define AT_AP6(N, _1, _2, _3, _4, _5, _6) AT_DISPATCH_CASE(_1, N) AT_DISPATCH_CASE(_2, N) AT_DISPATCH_CASE(_3, N) AT_DISPATCH_CASE(_4, N) AT_DISPATCH_CASE(_5, N) AT_DISPATCH_CASE(_6, N)
#define AT_AP7(N, _1, _2, _3, _4, _5, _6, _7) AT_DISPATCH_CASE(_1, N) AT_DISPATCH_CASE(_2, N) AT_DISPATCH_CASE(_3, N) AT_DISPATCH_CASE(_4, N) AT_DISPATCH_CASE(_5, N) AT_DISPATCH_CASE(_6, N) AT_DISPATCH_CASE(_7, N)
#define AT_AP8(N, _1, _2, _3, _4, _5, _6, _7, _8) AT_DISPATCH_CASE(_1, N) AT_DISPATCH_CASE(_2, N) AT_DISPATCH_CASE(_3, N) AT_DISPATCH_CASE(_4, N) AT_DISPATCH_CASE(_5, N) AT_DISPATCH_CASE(_6, N) AT_DISPATCH_CASE(_7, N) AT_DISPATCH_CASE(_8, N)
#define AT_AP9(N, _1, _2, _3, _4, _5, _6, _7, _8, _9) AT_DISPATCH_CASE(_1, N) AT_DISPATCH_CASE(_2, N) AT_DISPATCH_CASE(_3, N) AT_DISPATCH_CASE(_4, N) AT_DISPATCH_CASE(_5, N) AT_DISPATCH_CASE(_6, N) AT_DISPATCH_CASE(_7, N) AT_DISPATCH_CASE(_8, N) AT_DISPATCH_CASE(_9, N)
#define AT_AP10(N, _1, _2, _3, _4, _5, _6, _7, _8, _9, _10) AT_DISPATCH_CASE(_1, N) AT_DISPATCH_CASE(_2, N) AT_DISPATCH_CASE(_3, N) AT_DISPATCH_CASE(_4, N) AT_DISPATCH_CASE(_5, N) AT_DISPATCH_CASE(_6, N) AT_DISPATCH_CASE(_7, N) AT_DISPATCH_CASE(_8, N) AT_DISPATCH_CASE(_9, N) AT_DISPATCH_CASE(_10, N)
#define AT_AP11(N, _1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11) AT_DISPATCH_CASE(_1, N) AT_DISPATCH_CASE(_2, N) AT_DISPATCH_CASE(_3, N) AT_DISPATCH_CASE(_4, N) AT_DISPATCH_CASE(_5, N) AT_DISPATCH_CASE(_6, N) AT_DISPATCH_CASE(_7, N) AT_DISPATCH_CASE(_8, N) AT_DISPATCH_CASE(_9, N) AT_DISPATCH_CASE(_10, N) AT_DISPATCH_CASE(_11, N)
#define AT_AP12(N, _1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12) AT_DISPATCH_CASE(_1, N) AT_DISPATCH_CASE(_2, N) AT_DISPATCH_CASE(_3, N) AT_DISPATCH_CASE(_4, N) AT_DISPATCH_CASE(_5, N) AT_DISPATCH_CASE(_6, N) AT_DISPATCH_CASE(_7, N) AT_DISPATCH_CASE(_8, N) AT_DISPATCH_CASE(_9, N) AT_DISPATCH_CASE(_10, N) AT_DISPATCH_CASE(_11, N) AT_DISPATCH_CASE(_12, N)
#define AT_AP13(N, _1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13) AT_DISPATCH_CASE(_1, N) AT_DISPATCH_CASE(_2, N) AT_DISPATCH_CASE(_3, N) AT_DISPATCH_CASE(_4, N) AT_DISPATCH_CASE(_5, N) AT_DISPATCH_CASE(_6, N) AT_DISPATCH_CASE(_7, N) AT_DISPATCH_CASE(_8, N) AT_DISPATCH_CASE(_9, N) AT_DISPATCH_CASE(_10, N) AT_DISPATCH_CASE(_11, N) AT_DISPATCH_CASE(_12, N) AT_DISPATCH_CASE(_13, N)
#define AT_AP14(N, _1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14) AT_DISPATCH_CASE(_1, N) AT_DISPATCH_CASE(_2, N) AT_DISPATCH_CASE(_3, N) AT_DISPATCH_CASE(_4, N) AT_DISPATCH_CASE(_5, N) AT_DISPATCH_CASE(_6, N) AT_DISPATCH_CASE(_7, N) AT_DISPATCH_CASE(_8, N) AT_DISPATCH_CASE(_9, N) AT_DISPATCH_CASE(_10, N) AT_DISPATCH_CASE(_11, N) AT_DISPATCH_CASE(_12, N) AT_DISPATCH_CASE(_13, N) AT_DISPATCH_CASE(_14, N)
#define AT_AP15(N, _1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15) AT_DISPATCH_CASE(_1, N) AT_DISPATCH_CASE(_2, N) AT_DISPATCH_CASE(_3, N) AT_DISPATCH_CASE(_4, N) AT_DISPATCH_CASE(_5, N) AT_DISPATCH_CASE(_6, N) AT_DISPATCH_CASE(_7, N) AT_DISPATCH_CASE(_8, N) AT_DISPATCH_CASE(_9, N) AT_DISPATCH_CASE(_10, N) AT_DISPATCH_CASE(_11, N) AT_DISPATCH_CASE(_12, N) AT_DISPATCH_CASE(_13, N) AT_DISPATCH_CASE(_14, N) AT_DISPATCH_CASE(_15, N)
#define AT_AP16(N, _1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16) AT_DISPATCH_CASE(_1, N) AT_DISPATCH_CASE(_2, N) AT_DISPATCH_CASE(_3, N) AT_DISPATCH_CASE(_4, N) AT_DISPATCH_CASE(_5, N) AT_DISPATCH_CASE(_6, N) AT_DISPATCH_CASE(_7, N) AT_DISPATCH_CASE(_8, N) AT_DISPATCH_CASE(_9, N) AT_DISPATCH_CASE(_10, N) AT_DISPATCH_CASE(_11, N) AT_DISPATCH_CASE(_12, N) AT_DISPATCH_CASE(_13, N) AT_DISPATCH_CASE(_14, N) AT_DISPATCH_CASE(_15, N) AT_DISPATCH_CASE(_16, N)
#define AT_AP17(N, _1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16, _17) AT_DISPATCH_CASE(_1, N) AT_DISPATCH_CASE(_2, N) AT_DISPATCH_CASE(_3, N) AT_DISPATCH_CASE(_4, N) AT_DISPATCH_CASE(_5, N) AT_DISPATCH_CASE(_6, N) AT_DISPATCH_CASE(_7, N) AT_DISPATCH_CASE(_8, N) AT_DISPATCH_CASE(_9, N) AT_DISPATCH_CASE(_10, N) AT_DISPATCH_CASE(_11, N) AT_DISPATCH_CASE(_12, N) AT_DISPATCH_CASE(_13, N) AT_DISPATCH_CASE(_14, N) AT_DISPATCH_CASE(_15, N) AT_DISPATCH_CASE(_16, N) AT_DISPATCH_CASE(_17, N)
#define AT_AP18(N, _1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16, _17, _18) AT_DISPATCH_CASE(_1, N) AT_DISPATCH_CASE(_2, N) AT_DISPATCH_CASE(_3, N) AT_DISPATCH_CASE(_4, N) AT_DISPATCH_CASE(_5, N) AT_DISPATCH_CASE(_6, N) AT_DISPATCH_CASE(_7, N) AT_DISPATCH_CASE(_8, N) AT_DISPATCH_CASE(_9, N) AT_DISPATCH_CASE(_10, N) AT_DISPATCH_CASE(_11, N) AT_DISPATCH_CASE(_12, N) AT_DISPATCH_CASE(_13, N) AT_DISPATCH_CASE(_14, N) AT_DISPATCH_CASE(_15, N) AT_DISPATCH_CASE(_16, N) AT_DISPATCH_CASE(_17, N) AT_DISPATCH_CASE(_18, N)
#define AT_AP19(N, _1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16, _17, _18, _19) AT_DISPATCH_CASE(_1, N) AT_DISPATCH_CASE(_2, N) AT_DISPATCH_CASE(_3, N) AT_DISPATCH_CASE(_4, N) AT_DISPATCH_CASE(_5, N) AT_DISPATCH_CASE(_6, N) AT_DISPATCH_CASE(_7, N) AT_DISPATCH_CASE(_8, N) AT_DISPATCH_CASE(_9, N) AT_DISPATCH_CASE(_10, N) AT_DISPATCH_CASE(_11, N) AT_DISPATCH_CASE(_12, N) AT_DISPATCH_CASE(_13, N) AT_DISPATCH_CASE(_14, N) AT_DISPATCH_CASE(_15, N) AT_DISPATCH_CASE(_16, N) AT_DISPATCH_CASE(_17, N) AT_DISPATCH_CASE(_18, N) AT_DISPATCH_CASE(_19, N)
#define AT_AP20(N, _1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16, _17, _18, _19, _20) AT_DISPATCH_CASE(_1, N) AT_DISPATCH_CASE(_2, N) AT_DISPATCH_CASE(_3, N) AT_DISPATCH_CASE(_4, N) AT_DISPATCH_CASE(_5, N) AT_DISPATCH_CASE(_6, N) AT_DISPATCH_CASE(_7, N) AT_DISPATCH_CASE(_8, N) AT_DISPATCH_CASE(_9, N) AT_DISPATCH_CASE(_10, N) AT_DISPATCH_CASE(_11, N) AT_DISPATCH_CASE(_12, N) AT_DISPATCH_CASE(_13, N) AT_DISPATCH_CASE(_14, N) AT_DISPATCH_CASE(_15, N) AT_DISPATCH_CASE(_16, N) AT_DISPATCH_CASE(_17, N) AT_DISPATCH_CASE(_18, N) AT_DISPATCH_CASE(_19, N) AT_DISPATCH_CASE(_20, N)
#define AT_AP21(N, _1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16, _17, _18, _19, _20, _21) AT_DISPATCH_CASE(_1, N) AT_DISPATCH_CASE(_2, N) AT_DISPATCH_CASE(_3, N) AT_DISPATCH_CASE(_4, N) AT_DISPATCH_CASE(_5, N) AT_DISPATCH_CASE(_6, N) AT_DISPATCH_CASE(_7, N) AT_DISPATCH_CASE(_8, N) AT_DISPATCH_CASE(_9, N) AT_DISPATCH_CASE(_10, N) AT_DISPATCH_CASE(_11, N) AT_DISPATCH_CASE(_12, N) AT_DISPATCH_CASE(_13, N) AT_DISPATCH_CASE(_14, N) AT_DISPATCH_CASE(_15, N) AT_DISPATCH_CASE(_16, N) AT_DISPATCH_CASE(_17, N) AT_DISPATCH_CASE(_18, N) AT_DISPATCH_CASE(_19, N) AT_DISPATCH_CASE(_20, N) AT_DISPATCH_CASE(_21, N)
#define AT_AP22(N, _1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16, _17, _18, _19, _20, _21, _22) AT_DISPATCH_CASE(_1, N) AT_DISPATCH_CASE(_2, N) AT_DISPATCH_CASE(_3, N) AT_DISPATCH_CASE(_4, N) AT_DISPATCH_CASE(_5, N) AT_DISPATCH_CASE(_6, N) AT_DISPATCH_CASE(_7, N) AT_DISPATCH_CASE(_8, N) AT_DISPATCH_CASE(_9, N) AT_DISPATCH_CASE(_10, N) AT_DISPATCH_CASE(_11, N) AT_DISPATCH_CASE(_12, N) AT_DISPATCH_CASE(_13, N) AT_DISPATCH_CASE(_14, N) AT_DISPATCH_CASE(_15, N) AT_DISPATCH_CASE(_16, N) AT_DISPATCH_CASE(_17, N) AT_DISPATCH_CASE(_18, N) AT_DISPATCH_CASE(_19, N) AT_DISPATCH_CASE(_20, N) AT_DISPATCH_CASE(_21, N) AT_DISPATCH_CASE(_22, N)
#define AT_AP23(N, _1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16, _17, _18, _19, _20, _21, _22, _23) AT_DISPATCH_CASE(_1, N) AT_DISPATCH_CASE(_2, N) AT_DISPATCH_CASE(_3, N) AT_DISPATCH_CASE(_4, N) AT_DISPATCH_CASE(_5, N) AT_DISPATCH_CASE(_6, N) AT_DISPATCH_CASE(_7, N) AT_DISPATCH_CASE(_8, N) AT_DISPATCH_CASE(_9, N) AT_DISPATCH_CASE(_10, N) AT_DISPATCH_CASE(_11, N) AT_DISPATCH_CASE(_12, N) AT_DISPATCH_CASE(_13, N) AT_DISPATCH_CASE(_14, N) AT_DISPATCH_CASE(_15, N) AT_DISPATCH_CASE(_16, N) AT_DISPATCH_CASE(_17, N) AT_DISPATCH_CASE(_18, N) AT_DISPATCH_CASE(_19, N) AT_DISPATCH_CASE(_20, N) AT_DISPATCH_CASE(_21, N) AT_DISPATCH_CASE(_22, N) AT_DISPATCH_CASE(_23, N)
#define AT_AP24(N, _1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16, _17, _18, _19, _20, _21, _22, _23, _24) AT_DISPATCH_CASE(_1, N) AT_DISPATCH_CASE(_2, N) AT_DISPATCH_CASE(_3, N) AT_DISPATCH_CASE(_4, N) AT_DISPATCH_CASE(_5, N) AT_DISPATCH_CASE(_6, N) AT_DISPATCH_CASE(_7, N) AT_DISPATCH_CASE(_8, N) AT_DISPATCH_CASE(_9, N) AT_DISPATCH_CASE(_10, N) AT_DISPATCH_CASE(_11, N) AT_DISPATCH_CASE(_12, N) AT_DISPATCH_CASE(_13, N) AT_DISPATCH_CASE(_14, N) AT_DISPATCH_CASE(_15, N) AT_DISPATCH_CASE(_16, N) AT_DISPATCH_CASE(_17, N) AT_DISPATCH_CASE(_18, N) AT_DISPATCH_CASE(_19, N) AT_DISPATCH_CASE(_20, N) AT_DISPATCH_CASE(_21, N) AT_DISPATCH_CASE(_22, N) AT_DISPATCH_CASE(_23, N) AT_DISPATCH_CASE(_24, N)
#define AT_AP25(N, _1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16, _17, _18, _19, _20, _21, _22, _23, _24, _25) AT_DISPATCH_CASE(_1, N) AT_DISPATCH_CASE(_2, N) AT_DISPATCH_CASE(_3, N) AT_DISPATCH_CASE(_4, N) AT_DISPATCH_CASE(_5, N) AT_DISPATCH_CASE(_6, N) AT_DISPATCH_CASE(_7, N) AT_DISPATCH_CASE(_8, N) AT_DISPATCH_CASE(_9, N) AT_DISPATCH_CASE(_10, N) AT_DISPATCH_CASE(_11, N) AT_DISPATCH_CASE(_12, N) AT_DISPATCH_CASE(_13, N) AT_DISPATCH_CASE(_14, N) AT_DISPATCH_CASE(_15, N) AT_DISPATCH_CASE(_16, N) AT_DISPATCH_CASE(_17, N) AT_DISPATCH_CASE(_18, N) AT_DISPATCH_CASE(_19, N) AT_DISPATCH_CASE(_20, N) AT_DISPATCH_CASE(_21, N) AT_DISPATCH_CASE(_22, N) AT_DISPATCH_CASE(_23, N) AT_DISPATCH_CASE(_24, N) AT_DISPATCH_CASE(_25, N)
#define AT_AP26(N, _1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16, _17, _18, _19, _20, _21, _22, _23, _24, _25, _26) AT_DISPATCH_CASE(_1, N) AT_DISPATCH_CASE(_2, N) AT_DISPATCH_CASE(_3, N) AT_DISPATCH_CASE(_4, N) AT_DISPATCH_CASE(_5, N) AT_DISPATCH_CASE(_6, N) AT_DISPATCH_CASE(_7, N) AT_DISPATCH_CASE(_8, N) AT_DISPATCH_CASE(_9, N) AT_DISPATCH_CASE(_10, N) AT_DISPATCH_CASE(_11, N) AT_DISPATCH_CASE(_12, N) AT_DISPATCH_CASE(_13, N) AT_DISPATCH_CASE(_14, N) AT_DISPATCH_CASE(_15, N) AT_DISPATCH_CASE(_16, N) AT_DISPATCH_CASE(_17, N) AT_DISPATCH_CASE(_18, N) AT_DISPATCH_CASE(_19, N) AT_DISPATCH_CASE(_20, N) AT_DISPATCH_CASE(_21, N) AT_DISPATCH_CASE(_22, N) AT_DISPATCH_CASE(_23, N) AT_DISPATCH_CASE(_24, N) AT_DISPATCH_CASE(_25, N) AT_DISPATCH_CASE(_26, N)
#define AT_AP27(N, _1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16, _17, _18, _19, _20, _21, _22, _23, _24, _25, _26, _27) AT_DISPATCH_CASE(_1, N) AT_DISPATCH_CASE(_2, N) AT_DISPATCH_CASE(_3, N) AT_DISPATCH_CASE(_4, N) AT_DISPATCH_CASE(_5, N) AT_DISPATCH_CASE(_6, N) AT_DISPATCH_CASE(_7, N) AT_DISPATCH_CASE(_8, N) AT_DISPATCH_CASE(_9, N) AT_DISPATCH_CASE(_10, N) AT_DISPATCH_CASE(_11, N) AT_DISPATCH_CASE(_12, N) AT_DISPATCH_CASE(_13, N) AT_DISPATCH_CASE(_14, N) AT_DISPATCH_CASE(_15, N) AT_DISPATCH_CASE(_16, N) AT_DISPATCH_CASE(_17, N) AT_DISPATCH_CASE(_18, N) AT_DISPATCH_CASE(_19, N) AT_DISPATCH_CASE(_20, N) AT_DISPATCH_CASE(_21, N) AT_DISPATCH_CASE(_22, N) AT_DISPATCH_CASE(_23, N) AT_DISPATCH_CASE(_24, N) AT_DISPATCH_CASE(_25, N) AT_DISPATCH_CASE(_26, N) AT_DISPATCH_CASE(_27, N)
#define AT_AP28(N, _1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16, _17, _18, _19, _20, _21, _22, _23, _24, _25, _26, _27, _28) AT_DISPATCH_CASE(_1, N) AT_DISPATCH_CASE(_2, N) AT_DISPATCH_CASE(_3, N) AT_DISPATCH_CASE(_4, N) AT_DISPATCH_CASE(_5, N) AT_DISPATCH_CASE(_6, N) AT_DISPATCH_CASE(_7, N) AT_DISPATCH_CASE(_8, N) AT_DISPATCH_CASE(_9, N) AT_DISPATCH_CASE(_10, N) AT_DISPATCH_CASE(_11, N) AT_DISPATCH_CASE(_12, N) AT_DISPATCH_CASE(_13, N) AT_DISPATCH_CASE(_14, N) AT_DISPATCH_CASE(_15, N) AT_DISPATCH_CASE(_16, N) AT_DISPATCH_CASE(_17, N) AT_DISPATCH_CASE(_18, N) AT_DISPATCH_CASE(_19, N) AT_DISPATCH_CASE(_20, N) AT_DISPATCH_CASE(_21, N) AT_DISPATCH_CASE(_22, N) AT_DISPATCH_CASE(_23, N) AT_DISPATCH_CASE(_24, N) AT_DISPATCH_CASE(_25, N) AT_DISPATCH_CASE(_26, N) AT_DISPATCH_CASE(_27, N) AT_DISPATCH_CASE(_28, N)
#define AT_AP29(N, _1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16, _17, _18, _19, _20, _21, _22, _23, _24, _25, _26, _27, _28, _29) AT_DISPATCH_CASE(_1, N) AT_DISPATCH_CASE(_2, N) AT_DISPATCH_CASE(_3, N) AT_DISPATCH_CASE(_4, N) AT_DISPATCH_CASE(_5, N) AT_DISPATCH_CASE(_6, N) AT_DISPATCH_CASE(_7, N) AT_DISPATCH_CASE(_8, N) AT_DISPATCH_CASE(_9, N) AT_DISPATCH_CASE(_10, N) AT_DISPATCH_CASE(_11, N) AT_DISPATCH_CASE(_12, N) AT_DISPATCH_CASE(_13, N) AT_DISPATCH_CASE(_14, N) AT_DISPATCH_CASE(_15, N) AT_DISPATCH_CASE(_16, N) AT_DISPATCH_CASE(_17, N) AT_DISPATCH_CASE(_18, N) AT_DISPATCH_CASE(_19, N) AT_DISPATCH_CASE(_20, N) AT_DISPATCH_CASE(_21, N) AT_DISPATCH_CASE(_22, N) AT_DISPATCH_CASE(_23, N) AT_DISPATCH_CASE(_24, N) AT_DISPATCH_CASE(_25, N) AT_DISPATCH_CASE(_26, N) AT_DISPATCH_CASE(_27, N) AT_DISPATCH_CASE(_28, N) AT_DISPATCH_CASE(_29, N)
#define AT_AP30(N, _1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16, _17, _18, _19, _20, _21, _22, _23, _24, _25, _26, _27, _28, _29, _30) AT_DISPATCH_CASE(_1, N) AT_DISPATCH_CASE(_2, N) AT_DISPATCH_CASE(_3, N) AT_DISPATCH_CASE(_4, N) AT_DISPATCH_CASE(_5, N) AT_DISPATCH_CASE(_6, N) AT_DISPATCH_CASE(_7, N) AT_DISPATCH_CASE(_8, N) AT_DISPATCH_CASE(_9, N) AT_DISPATCH_CASE(_10, N) AT_DISPATCH_CASE(_11, N) AT_DISPATCH_CASE(_12, N) AT_DISPATCH_CASE(_13, N) AT_DISPATCH_CASE(_14, N) AT_DISPATCH_CASE(_15, N) AT_DISPATCH_CASE(_16, N) AT_DISPATCH_CASE(_17, N) AT_DISPATCH_CASE(_18, N) AT_DISPATCH_CASE(_19, N) AT_DISPATCH_CASE(_20, N) AT_DISPATCH_CASE(_21, N) AT_DISPATCH_CASE(_22, N) AT_DISPATCH_CASE(_23, N) AT_DISPATCH_CASE(_24, N) AT_DISPATCH_CASE(_25, N) AT_DISPATCH_CASE(_26, N) AT_DISPATCH_CASE(_27, N) AT_DISPATCH_CASE(_28, N) AT_DISPATCH_CASE(_29, N) AT_DISPATCH_CASE(_30, N)
#define AT_AP31(N, _1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16, _17, _18, _19, _20, _21, _22, _23, _24, _25, _26, _27, _28, _29, _30, _31) AT_DISPATCH_CASE(_1, N) AT_DISPATCH_CASE(_2, N) AT_DISPATCH_CASE(_3, N) AT_DISPATCH_CASE(_4, N) AT_DISPATCH_CASE(_5, N) AT_DISPATCH_CASE(_6, N) AT_DISPATCH_CASE(_7, N) AT_DISPATCH_CASE(_8, N) AT_DISPATCH_CASE(_9, N) AT_DISPATCH_CASE(_10, N) AT_DISPATCH_CASE(_11, N) AT_DISPATCH_CASE(_12, N) AT_DISPATCH_CASE(_13, N) AT_DISPATCH_CASE(_14, N) AT_DISPATCH_CASE(_15, N) AT_DISPATCH_CASE(_16, N) AT_DISPATCH_CASE(_17, N) AT_DISPATCH_CASE(_18, N) AT_DISPATCH_CASE(_19, N) AT_DISPATCH_CASE(_20, N) AT_DISPATCH_CASE(_21, N) AT_DISPATCH_CASE(_22, N) AT_DISPATCH_CASE(_23, N) AT_DISPATCH_CASE(_24, N) AT_DISPATCH_CASE(_25, N) AT_DISPATCH_CASE(_26, N) AT_DISPATCH_CASE(_27, N) AT_DISPATCH_CASE(_28, N) AT_DISPATCH_CASE(_29, N) AT_DISPATCH_CASE(_30, N) AT_DISPATCH_CASE(_31, N)
#define AT_AP32(N, _1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16, _17, _18, _19, _20, _21, _22, _23, _24, _25, _26, _27, _28, _29, _30, _31, _32) AT_DISPATCH_CASE(_1, N) AT_DISPATCH_CASE(_2, N) AT_DISPATCH_CASE(_3, N) AT_DISPATCH_CASE(_4, N) AT_DISPATCH_CASE(_5, N) AT_DISPATCH_CASE(_6, N) AT_DISPATCH_CASE(_7, N) AT_DISPATCH_CASE(_8, N) AT_DISPATCH_CASE(_9, N) AT_DISPATCH_CASE(_10, N) AT_DISPATCH_CASE(_11, N) AT_DISPATCH_CASE(_12, N) AT_DISPATCH_CASE(_13, N) AT_DISPATCH_CASE(_14, N) AT_DISPATCH_CASE(_15, N) AT_DISPATCH_CASE(_16, N) AT_DISPATCH_CASE(_17, N) AT_DISPATCH_CASE(_18, N) AT_DISPATCH_CASE(_19, N) AT_DISPATCH_CASE(_20, N) AT_DISPATCH_CASE(_21, N) AT_DISPATCH_CASE(_22, N) AT_DISPATCH_CASE(_23, N) AT_DISPATCH_CASE(_24, N) AT_DISPATCH_CASE(_25, N) AT_DISPATCH_CASE(_26, N) AT_DISPATCH_CASE(_27, N) AT_DISPATCH_CASE(_28, N) AT_DISPATCH_CASE(_29, N) AT_DISPATCH_CASE(_30, N) AT_DISPATCH_CASE(_31, N) AT_DISPATCH_CASE(_32, N)

// End generated code
// clang-format on