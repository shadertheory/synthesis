use std::{
    cell::RefCell,
    convert::identity,
    fmt,
    iter::Sum,
    mem::{self, MaybeUninit},
    ops::{BitAnd, BitOr, BitOrAssign, Deref, Index, IndexMut, Neg, Shl, Shr, *},
    slice,
};

pub trait Tensor {
    type Repr;
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct Matrix<const ROWS: usize, const COLS: usize, T>(pub [T; { ROWS * COLS }])
where
    [(); { ROWS * COLS }]: Sized;

macro_rules! matrix_elementwise_op {
    ($trait:tt, $func:tt, $op:tt, $op_assign:tt) => {
        impl<const ROWS: usize, const COLS: usize, T: $trait + Num> $trait<Self> for Matrix<ROWS, COLS, T>
        where
            <T as $trait>::Output: Num,
            [(); { ROWS * COLS }]: Sized,
        {
            type Output = Matrix<ROWS, COLS, <T as $trait>::Output>;

            fn $func(self, rhs: Self) -> <Self as $trait<Self>>::Output {
                let mut ret = <Self as $trait<Self>>::Output::identity();
                for i in 0..ROWS * COLS {
                    ret.0[i] = self.0[i] $op rhs.0[i];
                }
                ret
            }
        }

        paste::paste! {
            // Matrix op= Matrix
            impl<const ROWS: usize, const COLS: usize, T: $trait + [<$trait Assign>] + Num> [<$trait Assign>]<Self> for Matrix<ROWS, COLS, T>
            where
                <T as $trait>::Output: Num,
                [(); { ROWS * COLS }]: Sized,
            {
                fn [<$func _assign>](&mut self, rhs: Self) {
                    for i in 0..ROWS * COLS {
                        self.0[i] $op_assign rhs.0[i];
                    }
                }
            }
        }
    };
}

macro_rules! matrix_scalar_op {
    ($trait:tt, $func:tt, $op:tt, $op_assign:tt) => {

        impl<const ROWS: usize, const COLS: usize, T: $trait + Num> $trait<T> for Matrix<ROWS, COLS, T>
        where
            <T as $trait>::Output: Num,
            [(); { ROWS * COLS }]: Sized,
        {
            type Output = Matrix<ROWS, COLS, <T as $trait>::Output>;

            fn $func(self, scalar: T) -> <Self as $trait<T>>::Output {
                let mut ret = <Self as $trait<T>>::Output::identity();
                for i in 0..ROWS * COLS {
                    ret.0[i] = self.0[i] $op scalar;
                }
                ret
            }
        }

        paste::paste! {
            // Matrix op= scalar
            impl<const ROWS: usize, const COLS: usize, T: $trait + [<$trait Assign>] + Num> [<$trait Assign>]<T> for Matrix<ROWS, COLS, T>
            where
                <T as $trait>::Output: Num,
                [(); { ROWS * COLS }]: Sized,
            {
                fn [<$func _assign>](&mut self, scalar: T) {
                    for i in 0..ROWS * COLS {
                        self.0[i] $op_assign scalar;
                    }
                }
            }
        }
    };
}

// Apply macros
matrix_elementwise_op!(Add, add, +, +=);
matrix_elementwise_op!(Sub, sub, -, -=);
matrix_scalar_op!(Mul, mul, *, *=);
matrix_scalar_op!(Div, div, /, /=);

// Negation (unary operator)
impl<const ROWS: usize, const COLS: usize, T> Neg for Matrix<ROWS, COLS, T>
where
    T: Neg + Num,
    <T as Neg>::Output: Num,
    [(); { ROWS * COLS }]: Sized,
{
    type Output = Matrix<ROWS, COLS, <T as Neg>::Output>;

    fn neg(self) -> <Self as Neg>::Output {
        let mut result = <Self as Neg>::Output::identity();
        for i in 0..ROWS * COLS {
            result.0[i] = -self.0[i];
        }
        result
    }
}
impl<T: Float> Mul<Vector<3, T>> for Quaternion<T> {
    type Output = Vector<3, T>;
    fn mul(self, rhs: Vector<3, T>) -> <Self as Mul<Vector<3, T>>>::Output {
        let Quaternion {
            scalar: w,
            vector: u,
        } = self;
        let v = rhs;
        let two = T::ONE + T::ONE;
        u * (two * u.dot(v)) + v * (w * w - u.magnitude_squared()) + u.cross(v) * (two * w)
    }
}
// Matrix multiplication (special case - can't use macro)
impl<const ROWS: usize, const COLS: usize, T> Mul<Vector<COLS, T>> for Matrix<ROWS, COLS, T>
where
    T: Num,
    [(); { ROWS * COLS }]: Sized,
    [(); { ROWS }]: Sized,
    [(); { COLS }]: Sized,
{
    type Output = Vector<COLS, T>;

    fn mul(self, rhs: Vector<COLS, T>) -> <Self as Mul<Vector<COLS, T>>>::Output {
        let mut result = Vector::ZERO;

        for row in 0..ROWS {
            let mut sum = T::ZERO;
            for col in 0..COLS {
                sum = sum + self[(row, col)] * rhs[col];
            }
            result[row] = sum;
        }

        result
    }
}
impl<const M: usize, const N: usize, const P: usize, T> Mul<Matrix<N, P, T>> for Matrix<M, N, T>
where
    T: Float,
    [(); { M * N }]: Sized,
    [(); { N * P }]: Sized,
    [(); { M * P }]: Sized,
{
    type Output = Matrix<M, P, T>;

    fn mul(self, rhs: Matrix<N, P, T>) -> <Self as Mul<Matrix<N, P, T>>>::Output {
        let mut result = [T::ZERO; M * P];

        for i in 0..M {
            // For each row i
            for j in 0..P {
                // For each col j
                let mut sum = T::ZERO;
                for k in 0..N {
                    // For each common dim k
                    let self_val = self.0[k * M + i]; // self[i,k]
                    let rhs_val = rhs.0[j * N + k]; // rhs[k,j]
                    sum = sum + self_val * rhs_val;
                }
                result[j * M + i] = sum;
            }
        }
        Matrix(result)
    }
}

impl<const ROWS: usize, const COLS: usize, T: Num> FromIterator<T> for Matrix<ROWS, COLS, T>
where
    [(); { ROWS * COLS }]: Sized,
{
    fn from_iter<IntoIter: IntoIterator<Item = T>>(iter: IntoIter) -> Self {
        let mut matrix = Self::identity();
        for (index, num) in (0..ROWS)
            .flat_map(|row| (0..COLS).map(move |col| (row, col)))
            .zip(iter)
        {
            matrix[index] = num;
        }
        matrix
    }
}

impl<const ROWS: usize, const COLS: usize, T: Num> Matrix<ROWS, COLS, T>
where
    [(); { ROWS * COLS }]: Sized,
{
    pub const ZERO: Self = Self([T::ZERO; { ROWS * COLS }]);
    pub fn identity() -> Self {
        let mut arr = [T::ZERO; { ROWS * COLS }];
        for index in 0..ROWS * COLS {
            if index % (ROWS + 1) == 0 {
                arr[index] = T::ONE;
            };
        }
        Self(arr)
    }

    pub fn from_translation(translation: Vector<{ COLS - 1 }, T>) -> Self {
        let mut matrix = Matrix::identity();
        for row in 0..ROWS - 1 {
            let dim = row;
            matrix[(row, COLS - 1)] = translation[dim];
        }
        matrix
    }

    pub fn transpose(self) -> Self {
        (0..ROWS)
            .flat_map(|row| (0..COLS).map(move |col| (row, col)))
            .map(|(row, col)| self[(col, row)])
            .collect()
    }
    pub fn minor(self, skip_row: usize, skip_col: usize) -> Matrix<{ ROWS - 1 }, { COLS - 1 }, T>
    where
        [(); { (ROWS - 1) * (COLS - 1) }]: Sized,
    {
        (0..ROWS)
            .flat_map(|row| (0..COLS).map(move |col| (row, col)))
            .filter(|&(row, col)| row != skip_row && col != skip_col)
            .map(|index| self[index])
            .collect()
    }

    pub fn from_array(array: [T; { ROWS * COLS }]) -> Self {
        Self(array)
    }
    pub fn to_array(self) -> [T; { ROWS * COLS }]
    where
        [(); { ROWS * COLS }]: Sized,
    {
        self.0
    }

    pub fn determinant(&self) -> T
    where
        T: Signed,
    {
        pub fn two_by_two<T: Num>(this: Matrix<2, 2, T>) -> T {
            this[(0, 0)] * this[(1, 1)] - this[(0, 1)] * this[(1, 0)]
        }
        if ROWS == 2 && COLS == 2 {
            return (two_by_two)(self.to_array().into_iter().collect());
        }
        if ROWS != COLS {
            panic!("determinant can only be taken on square matrices");
        }
        let mut stack = vec![(self.to_array().to_vec(), ROWS, T::ONE)];
        let mut result = T::ZERO;
        while let Some((matrix, num_rows, coef)) = stack.pop() {
            let num_cols = num_rows;
            if num_rows == 2 {
                result += coef * two_by_two(matrix.into_iter().collect());
            } else {
                for col in 0..num_rows {
                    let element = matrix[col];
                    let sign = if col % 2 == 0 { T::ONE } else { T::ONE.neg() };

                    let mut minor = vec![];
                    for row in 1..num_rows {
                        for skip_col in 0..num_rows {
                            if skip_col != col {
                                let col = skip_col;
                                minor.push(matrix[row * num_cols + col]);
                            }
                        }
                    }

                    stack.push((minor, num_rows - 1, coef * sign * element))
                }
            }
        }
        result
    }
    pub fn inverse(self) -> Option<Self>
    where
        [(); (ROWS - 1) * (COLS - 1)]: Sized,
        T: Signed + fmt::Debug,
    {
        if ROWS != COLS {
            panic!("inverse can only be taken on square matrices");
        }

        let det = self.determinant();

        if det == T::ZERO {
            return None;
        }

        if ROWS == 2 {
            let mut result = Self::identity();
            result[(0, 0)] = self[(1, 1)] / det;
            result[(0, 1)] = (-T::ONE * self[(0, 1)]) / det;
            result[(1, 0)] = (-T::ONE * self[(1, 0)]) / det;
            result[(1, 1)] = self[(0, 0)] / det;
            return Some(result);
        }

        let mut cofactors = Self::identity();

        for (row, col) in (0..ROWS).flat_map(|row| (0..COLS).map(move |col| (row, col))) {
            let minor_det = self.minor(row, col).determinant();

            let sign = if (row + col) % 2 == 0 {
                T::ONE
            } else {
                -T::ONE
            };
            cofactors[(row, col)] = sign * minor_det;
        }

        let adjugate = cofactors.transpose();

        let mut result = Self::identity();
        for index in (0..ROWS).flat_map(|row| (0..COLS).map(move |col| (row, col))) {
            result[index] = adjugate[index] / det;
        }
        Some(result)
    }
}

type RowCol = (usize, usize);
impl<const ROWS: usize, const COLS: usize, T> Index<RowCol> for Matrix<ROWS, COLS, T>
where
    [T; { ROWS * COLS }]: Sized,
{
    type Output = T;

    fn index(&self, index: RowCol) -> &<Self as Index<RowCol>>::Output {
        let (row, col) = index;
        &self.0[col * ROWS + row]
    }
}

impl<const ROWS: usize, const COLS: usize, T> IndexMut<RowCol> for Matrix<ROWS, COLS, T>
where
    [T; { ROWS * COLS }]: Sized,
{
    fn index_mut(&mut self, index: RowCol) -> &mut <Self as Index<RowCol>>::Output {
        let (row, col) = index;
        &mut self.0[col * ROWS + row]
    }
}

#[derive(Clone, Copy)]
pub struct Projection {
    pub near: f32,
    pub fov: f32,
    pub aspect: f32,
}

impl Projection {
    pub fn to_matrix(self) -> Matrix<4, 4, f32> {
        let angle = (self.fov / 2.0);
        use crate::Trig;
        let f = 1.0 / (self.fov / 2.0).cot();
        let mut matrix = Matrix::ZERO;

        // Column 0
        matrix[(0, 0)] = f / self.aspect;

        // oolumn 1
        matrix[(1, 1)] = f;

        // Column 2
        matrix[(3, 2)] = self.near;

        // Column 3
        matrix[(2, 3)] = -1.0;

        matrix
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Vector<const DIM: usize, T>(pub [T; DIM]);

#[derive(Debug, Clone, Copy)]
pub struct Raycast<const DIM: usize, T> {
    pub origin: Vector<DIM, T>,
    pub direction: Vector<DIM, T>,
}

impl<const DIM: usize, T: Num> Raycast<DIM, T> {
    fn at(self, distance: T) -> Vector<DIM, T> {
        self.origin + self.direction * distance
    }
}

pub trait Intersect<const DIM: usize> {
    type Scalar;
    fn intersect(
        &self,
        ray: Raycast<DIM, <Self as Intersect<DIM>>::Scalar>,
    ) -> Option<Vector<DIM, <Self as Intersect<DIM>>::Scalar>>;
}

pub struct Plane {
    pub origin: Vector<3, f32>,
    pub normal: Vector<3, f32>,
}

impl Plane {
    pub const Z: Self = Self {
        origin: Vector::ZERO,
        normal: Vector::<3, f32>::Z,
    };
}

impl Intersect<3> for Plane {
    type Scalar = f32;

    fn intersect(
        &self,
        ray: Raycast<3, <Self as Intersect<3>>::Scalar>,
    ) -> Option<Vector<3, <Self as Intersect<3>>::Scalar>> {
        let denominator = (0..3).fold(0.0, |accum, i| accum + self.normal[i] * ray.direction[i]);

        if denominator == 0.0 {
            None?
        }

        let numerator = (0..3).fold(0.0, |accum, i| {
            accum + self.normal[i] * (self.origin[i] - ray.origin[i])
        });

        let distance = numerator / denominator;

        if distance < 0.0 {
            None?
        }

        Some(ray.at(distance))
    }
}

impl<const DIM: usize, T> Tensor for Vector<DIM, T> {
    type Repr = T;
}

impl<const DIM: usize, T: Num> Vector<DIM, T> {
    pub const ZERO: Self = Self([T::ZERO; DIM]);

    fn to_array(self) -> [T; DIM] {
        self.0
    }
}

impl<const DIM: usize, T> Index<usize> for Vector<DIM, T> {
    type Output = T;
    fn index(&self, index: usize) -> &<Self as Index<usize>>::Output {
        &self.0[index]
    }
}

impl<const DIM: usize, T> IndexMut<usize> for Vector<DIM, T> {
    fn index_mut(&mut self, index: usize) -> &mut <Self as Index<usize>>::Output {
        &mut self.0[index]
    }
}

pub trait Zero {
    const ZERO: Self;
}
pub trait One {
    const ONE: Self;
}

macro_rules! primitive {
    ($type:tt, $zero:tt, $one:tt, $pow:tt) => {
        impl Zero for $type {
            const ZERO: Self = $zero;
        }

        impl One for $type {
            const ONE: Self = $one;
        }

        impl Pow for $type {
            fn pow(self, exponent: i32) -> Self {
                $type::$pow(self, exponent as _)
            }
        }
    };
}
macro_rules! float {
    ($type:tt, $pi:expr) => {
        impl Trig for $type {
            const PI: Self = $pi;
            fn cos(self) -> Self {
                $type::cos(self)
            }
            fn arc_cos(self) -> Self {
                $type::acos(self)
            }
            fn sin(self) -> Self {
                $type::sin(self)
            }
            fn arc_sin(self) -> Self {
                $type::asin(self)
            }
            fn tan(self) -> Self {
                $type::tan(self)
            }
            fn cot(self) -> Self {
                $type::cos(self) / $type::sin(self)
            }
            fn arc_tan(self) -> Self {
                $type::atan(self)
            }
            fn arc_tan2(self, other: Self) -> Self {
                $type::atan2(self, other)
            }
        }
        impl Epsilon for $type {
            const EPSILON: $type = 0.000001;
        }
        impl Sqrt for $type {
            fn sqrt(self) -> Self {
                self.sqrt()
            }
        }
    };
}
primitive!(usize, 0, 1, pow);
primitive!(u32, 0, 1, pow);
primitive!(f32, 0.0, 1.0, powi);
float!(f32, 3.1415926535);

#[derive(Clone, Copy, Debug)]
pub struct Quaternion<T> {
    pub scalar: T,
    pub vector: Vector<3, T>,
}
impl<T: Signed> Neg for Quaternion<T> {
    type Output = Self;

    fn neg(self) -> <Self as Neg>::Output {
        self.conjugate()
    }
}
impl<T: Float> Quaternion<T> {
    /// Creates a rotation that looks in the given direction with the specified up vector
    /// For Z-up coordinate system
    pub fn look_rotation(forward: Vector<3, T>, up: Vector<3, T>) -> Self {
        let forward = forward.normalize();

        // Build orthonormal basis
        // In Z-up: forward is the look direction, up is Z axis typically
        let vector = forward.normalize();
        let vector2 = up.cross(vector).normalize(); // right vector
        let vector3 = vector.cross(vector2); // corrected up vector

        // Build rotation matrix (column-major to match your Matrix implementation)
        // Column 0: right (vector2)
        let m00 = vector2[0];
        let m10 = vector2[1];
        let m20 = vector2[2];

        // Column 1: up (vector3)
        let m01 = vector3[0];
        let m11 = vector3[1];
        let m21 = vector3[2];

        // Column 2: forward (vector)
        let m02 = vector[0];
        let m12 = vector[1];
        let m22 = vector[2];

        // Convert rotation matrix to quaternion
        let trace = m00 + m11 + m22;
        let one = T::ONE;
        let half = one / (one + one);

        if trace > T::ZERO {
            let s = (trace + one).sqrt();
            let w = s * half;
            let s = half / s;
            let x = (m21 - m12) * s;
            let y = (m02 - m20) * s;
            let z = (m10 - m01) * s;

            Quaternion {
                scalar: w,
                vector: Vector([x, y, z]),
            }
        } else if (m00 >= m11) && (m00 >= m22) {
            let s = ((one + m00) - m11 - m22).sqrt();
            let inv_s = half / s;
            let x = half * s;
            let y = (m10 + m01) * inv_s;
            let z = (m20 + m02) * inv_s;
            let w = (m21 - m12) * inv_s;

            Quaternion {
                scalar: w,
                vector: Vector([x, y, z]),
            }
        } else if m11 > m22 {
            let s = ((one + m11) - m00 - m22).sqrt();
            let inv_s = half / s;
            let x = (m01 + m10) * inv_s;
            let y = half * s;
            let z = (m12 + m21) * inv_s;
            let w = (m02 - m20) * inv_s;

            Quaternion {
                scalar: w,
                vector: Vector([x, y, z]),
            }
        } else {
            let s = ((one + m22) - m00 - m11).sqrt();
            let inv_s = half / s;
            let x = (m02 + m20) * inv_s;
            let y = (m12 + m21) * inv_s;
            let z = half * s;
            let w = (m10 - m01) * inv_s;

            Quaternion {
                scalar: w,
                vector: Vector([x, y, z]),
            }
        }
    }
}
impl<T: Num> Quaternion<T> {
    pub const IDENTITY: Self = Self {
        scalar: T::ONE,
        vector: Vector::<3, T>::ZERO,
    };

    pub fn to_matrix(mut self) -> Matrix<4, 4, T>
    where
        T: Sqrt + Epsilon,
    {
        let Self {
            vector: Vector([x, y, z]),
            scalar: w,
        } = self.normalize();
        let one = T::ONE;
        let two = one + one;
        let zero = T::ZERO;
        Matrix::from_array([
            // Column 0 (right vector)
            one - two * y * y - two * z * z,
            two * x * y + two * z * w,
            two * x * z - two * y * w,
            zero,
            // Column 1 (up vector)
            two * x * y - two * z * w,
            one - two * x * x - two * z * z,
            two * y * z + two * x * w,
            zero,
            // Column 2 (forward vector)
            two * x * z + two * y * w,
            two * y * z - two * x * w,
            one - two * x * x - two * y * y,
            zero,
            // Column 3 (translation)
            zero,
            zero,
            zero,
            one,
        ])
    }
    pub fn from_direction(direction: Vector<3, T>, up: Vector<3, T>) -> Self
    where
        T: Float + Trig,
    {
        let dir = direction.normalize();

        // Calculate yaw (rotation around Z axis)
        let yaw = dir[1].arc_tan2(dir[0]);

        // Calculate pitch (rotation in the vertical plane)
        let horizontal_length = (dir[0] * dir[0] + dir[1] * dir[1]).sqrt();
        let pitch = dir[2].arc_tan2(horizontal_length);

        // Create quaternions for yaw and pitch.
        let q_yaw = Quaternion::from_axis_angle(
            Vector([T::ZERO, T::ZERO, T::ONE]), // Z-up axis
            yaw,
        );

        let q_pitch = Quaternion::from_axis_angle(
            Vector([-T::ONE, T::ZERO, T::ZERO]), // X axis (negative for proper pitch)
            pitch,
        );

        // Combine: yaw first, then pitch
        q_yaw * q_pitch
    }
    pub fn conjugate(mut self) -> Self
    where
        T: Signed,
    {
        self.vector = -self.vector;
        self
    }

    pub fn normalize(mut self) -> Self
    where
        T: Sqrt + Epsilon,
    {
        let mag = (self.scalar * self.scalar + self.vector.magnitude_squared()).sqrt();
        if mag > T::EPSILON {
            // A real T::EPSILON would be better
            self.scalar = self.scalar / mag;
            self.vector = self.vector / mag;
        }
        self
    }
}

impl<T: Float> Quaternion<T> {
    pub fn from_axis_angle(axis: Vector<3, T>, angle: T) -> Self {
        let half = T::ONE / (T::ONE + T::ONE);
        let half_angle = angle * half;
        Self {
            scalar: half_angle.cos(),
            vector: axis.normalize() * half_angle.sin(),
        }
    }
}

impl<T: Num> Mul<Self> for Quaternion<T>
where
    T: Signed,
{
    type Output = Self;

    fn mul(self, rhs: Self) -> <Self as Mul<Self>>::Output {
        let Quaternion {
            scalar: w1,
            vector: Vector([x1, y1, z1]),
        } = self;
        let Quaternion {
            scalar: w2,
            vector: Vector([x2, y2, z2]),
        } = rhs;

        Self {
            scalar: w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            vector: Vector([
                w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            ]),
        }
    }
}

pub trait Pow {
    fn pow(self, exponent: i32) -> Self;
}

pub trait Epsilon: Sized {
    const EPSILON: Self;
}
pub trait Abs {
    fn abs(self) -> Self;
}
impl Abs for f32 {
    fn abs(self) -> Self {
        f32::abs(self)
    }
}
pub trait Trig: Sized {
    const PI: Self;
    fn cos(self) -> Self;
    fn arc_cos(self) -> Self;
    fn sin(self) -> Self;
    fn arc_sin(self) -> Self;
    fn tan(self) -> Self;
    fn cot(self) -> Self;
    fn arc_tan(self) -> Self;
    fn arc_tan2(self, rhs: Self) -> Self;
}

pub trait Sqrt {
    fn sqrt(self) -> Self;
}

pub trait Math = Sized
    + Add<Output = Self>
    + AddAssign
    + Sub<Output = Self>
    + SubAssign
    + Mul<Output = Self>
    + MulAssign
    + Div<Output = Self>
    + DivAssign
    + Rem<Output = Self>
    + Pow
    + Sum
    + PartialEq
    + PartialOrd;
pub trait Num = Copy + Zero + One + Math;
pub trait Signed = Num + Abs + Neg<Output = Self>;
pub trait Float = Num + Trig + Sqrt + Epsilon + Signed;

impl<const DIM: usize, T: Num> Default for Vector<DIM, T> {
    fn default() -> Self {
        let mut ret = unsafe { mem::zeroed::<[T; DIM]>() };
        for i in 0..DIM {
            ret[i] = T::ZERO;
        }
        Self(ret)
    }
}

macro_rules! vector_op {
    ($trait:tt, $func:tt, $op:tt, $op_assign:tt) => {
        use std::ops::$trait;
        impl<const DIM: usize, T: $trait + Num> $trait<Self> for Vector<DIM, T> where <T as $trait>::Output: Num {
            type Output = Vector<DIM, <T as $trait>::Output>;
            fn $func(self, rhs: Self) -> <Self as $trait<Self>>::Output {
                let mut ret = <Self as $trait<Self>>::Output::default();
                for i in 0..DIM {
                    ret[i] = self[i] $op rhs[i];
                }
                ret
            }
        }
        impl<const DIM: usize, T: $trait + Num> $trait<T> for Vector<DIM, T> where <T as $trait>::Output: Num {
            type Output = Vector<DIM, <T as $trait>::Output>;
            fn $func(self, rhs: T) -> <Self as $trait<T>>::Output {
                let mut ret = <Self as $trait<T>>::Output::default();
                for i in 0..DIM {
                    ret[i] = self[i] $op rhs;
                }
                ret
            }
        }
        paste::paste! {
            use std::ops::[<$trait Assign>];

            impl<const DIM: usize, T: $trait + [<$trait Assign>] + Num> [<$trait Assign>]<Self> for Vector<DIM, T> where <T as $trait>::Output: Num {
                fn [<$func _assign>](&mut self, rhs: Self) {
                    for i in 0..DIM {
                        self[i] $op_assign rhs[i];
                    }
                }
            }
            impl<const DIM: usize, T: $trait + [<$trait Assign>] + Num> [<$trait Assign>]<T> for Vector<DIM, T> where <T as $trait>::Output: Num {
                fn [<$func _assign>](&mut self, rhs: T) {
                    for i in 0..DIM {
                        self[i] $op_assign rhs;
                    }
                }
            }
        }
    };
}

impl<T: Num> Vector<2, T> {
    pub const X: Self = Self([T::ONE, T::ZERO]);
    pub const Y: Self = Self([T::ZERO, T::ONE]);
}
impl<T: Num> Vector<3, T> {
    pub const X: Self = Self([T::ONE, T::ZERO, T::ZERO]);
    pub const Y: Self = Self([T::ZERO, T::ONE, T::ZERO]);
    pub const Z: Self = Self([T::ZERO, T::ZERO, T::ONE]);
}
impl<T: Num> Vector<4, T> {
    pub const X: Self = Self([T::ONE, T::ZERO, T::ZERO, T::ZERO]);
    pub const Y: Self = Self([T::ZERO, T::ONE, T::ZERO, T::ZERO]);
    pub const Z: Self = Self([T::ZERO, T::ZERO, T::ONE, T::ZERO]);
    pub const W: Self = Self([T::ZERO, T::ZERO, T::ZERO, T::ONE]);

    pub fn perspective_divide(self) -> Vector<3, T> {
        Vector([self[0] / self[3], self[1] / self[3], self[2] / self[3]])
    }
}
impl<const DIM: usize, T: Num> Vector<DIM, T> {
    pub fn dot(self, rhs: Self) -> T {
        self.to_array()
            .into_iter()
            .zip(rhs.to_array())
            .fold(T::ZERO, |accum, (lhs, rhs)| accum + lhs * rhs)
    }
    pub fn cross(self, rhs: Self) -> Self
    where
        T: Signed,
    {
        assert!(DIM == 3, "Cross product is only defined for 3D vectors");

        Self(std::array::from_fn(|i| {
            let j = (i + 1) % 3;
            let k = (i + 2) % 3;
            self[j] * rhs[k] - self[k] * rhs[j]
        }))
    }
    pub fn magnitude_squared(self) -> T {
        self.dot(self)
    }
    pub fn magnitude(self) -> T
    where
        T: Sqrt,
    {
        self.magnitude_squared().sqrt()
    }
    pub fn normalize(self) -> Self
    where
        T: Sqrt,
    {
        self / self.magnitude()
    }
}

impl<const DIM: usize, T: Copy + Neg<Output = T>> Neg for Vector<DIM, T> {
    type Output = Self;
    fn neg(mut self) -> <Self as Neg>::Output {
        for i in 0..DIM {
            self[i] = -(self[i]);
        }
        self
    }
}

vector_op!(Add, add, +, +=);
vector_op!(Sub, sub, -, -=);
vector_op!(Mul, mul, *, *=);
vector_op!(Div, div, /, /=);
vector_op!(Rem, rem, %, %=);

pub trait IndexExt<const DIM: usize>: Tensor {
    fn to_one_dim(self, size: <Self as Tensor>::Repr) -> <Self as Tensor>::Repr;
    fn from_one_dim(idx: <Self as Tensor>::Repr, size: <Self as Tensor>::Repr) -> Self;
}

impl<T: Num> IndexExt<2> for Vector<2, T> {
    fn to_one_dim(self, size: T) -> <Self as Tensor>::Repr {
        let Self([x, y]) = self;
        x + y * size
    }

    fn from_one_dim(idx: <Self as Tensor>::Repr, size: T) -> Self {
        Self([idx % size, idx / size])
    }
}

impl<T: Num> IndexExt<3> for Vector<3, T> {
    fn to_one_dim(self, size: T) -> <Self as Tensor>::Repr {
        let Self([x, y, z]) = self;
        x + size * (y + z * size)
    }
    fn from_one_dim(idx: <Self as Tensor>::Repr, size: T) -> Self {
        let plane_size = size * size;
        let z = idx / plane_size;
        let plane = idx - z * plane_size;
        Vector([plane % size, plane / size, z])
    }
}
