from collections.abc import Callable
from sympy.core.basic import Basic
from sympy.core.cache import cacheit
from sympy.core import S, Dummy, Lambda, factor_terms
from sympy.core.symbol import Str, Symbol, symbols
from sympy.core.sympify import sympify
from sympy.functions.elementary.trigonometric import TrigonometricFunction
from sympy.logic.boolalg import BooleanTrue, BooleanFalse
from sympy.matrices.immutable import ImmutableDenseMatrix as Matrix
from sympy.matrices.matrixbase import MatrixBase
from sympy.solvers import solve
from sympy.vector.scalar import BaseScalar, BaseScalarFuncOfTime
from sympy.core.containers import Tuple
from sympy.core.function import diff
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (acos, atan2, cos, sin)
from sympy.matrices.dense import eye
from sympy.matrices.immutable import ImmutableDenseMatrix
from sympy.simplify.simplify import simplify
from sympy.simplify.fu import TR5
from sympy.simplify.trigsimp import trigsimp
import sympy.vector
from sympy.vector.orienters import (Orienter, AxisOrienter, BodyOrienter,
                                    SpaceOrienter, QuaternionOrienter)
from sympy.vector.coordsys_templates import (
    _get_coordsys_template, coordsys_registry)


class CoordSys3D(Basic):
    """
    Represents a coordinate system in 3-D space.
    """

    def __new__(
        cls, name, transformation=None, parent=None, location=None,
        rotation_matrix=None, vector_names=None, variable_names=None,
        time_symbol=None
    ):
        """
        The orientation/location parameters are necessary if this system
        is being defined at a certain orientation or location wrt another.

        Parameters
        ==========

        name : str
            The name of the new CoordSys3D instance.

        transformation : Lambda, Tuple, str
            Transformation defined by transformation equations or chosen
            from predefined ones.

        location : Vector
            The position vector of the new system's origin wrt the parent
            instance.

        rotation_matrix : SymPy ImmutableMatrix
            The rotation matrix of the new coordinate system with respect
            to the parent. In other words, the output of
            new_system.rotation_matrix(parent).

        parent : CoordSys3D
            The coordinate system wrt which the orientation/location
            (or both) is being defined.

        vector_names, variable_names : iterable(optional)
            Iterables of 3 strings each, with custom names for base
            vectors and base scalars of the new system respectively.
            Used for simple str printing.

        """
        args = (name, transformation, parent, location,
            rotation_matrix, vector_names, variable_names,
            time_symbol)

        if not cls._is_rebuilding(*args):
            # when user creates a new instance, args need to be processed
            args = cls._process_input_args(*args)

        return cls._create_new(*args)

    @classmethod
    def _is_rebuilding(
        cls, name, transformation, parent, origin,
        rotation_matrix, vector_names, variable_names, time_symbol
    ):
        """
        Check if the parameters are of the correct type. If this function
        returns False, then the arguments must be processed before creating
        the new object.
        """
        check_names = lambda v: (
            isinstance(v, Tuple) and (len(v) == 3)
            and all(isinstance(t, Str) for t in v))

        return (
            isinstance(name, Str)
            and isinstance(transformation, (Str, Lambda))
            and isinstance(parent, (cls, BooleanFalse))
            and isinstance(origin, sympy.vector.Point)
            and isinstance(rotation_matrix, ImmutableDenseMatrix)
            and isinstance(time_symbol, (Symbol, BooleanFalse))
            and check_names(vector_names)
            and check_names(variable_names)
        )

    @classmethod
    def _create_new(
        cls, name, transformation, parent, origin,
        rotation_matrix, vector_names, variable_names, time_symbol
    ):
        system = super().__new__(
            cls, name, transformation, parent, origin,
            rotation_matrix, vector_names, variable_names, time_symbol)
        cls._create_base_vectors(system, vector_names)
        cls._create_base_scalars(system, variable_names, time_symbol)
        cls._set_public_scalar_vector_attributes(
            system, vector_names, variable_names)
        cls._set_transformation_private_attributes(
            system, parent, origin, transformation, rotation_matrix)
        return system

    @property
    def root(self):
        if isinstance(self.parent, self.func):
            return self.parent.root
        return self

    @property
    def name(self):
        return self.args[0].name

    @property
    def parent(self):
        if not isinstance(self.args[2], self.func):
            return None
        return self.args[2]

    @property
    def variable_names(self):
        return tuple(s.name for s in self.args[-2])

    @property
    def vector_names(self):
        return tuple(s.name for s in self.args[-3])

    @property
    def transformation(self):
        if isinstance(self.args[1], Str):
            return self.args[1].name
        return self.args[1]

    @staticmethod
    def _create_base_vectors(system, vector_names):
        base_vectors = [
            BaseVector(i, system, name, name)
            for i, name in enumerate(vector_names)]
        system._base_vectors = tuple(base_vectors)

    @staticmethod
    def _create_base_scalars(system, variable_names, time_symbol):
        if time_symbol:
            base_scalars = [
                BaseScalarFuncOfTime(i, system, time_symbol)
                for i, name in enumerate(variable_names)]
        else:
            base_scalars = [
                BaseScalar(i, system, name, name)
                for i, name in enumerate(variable_names)]
        system._base_scalars = tuple(base_scalars)

    @staticmethod
    def _set_public_scalar_vector_attributes(
        system, vector_names, variable_names
    ):
        # TODO: this design choice is so wrong:
        # 1. user can chose any name for the attributes, even ones that can't
        #    be accessed.
        # 2. user can set any value to these attributes,
        #    clusterfucking everything.
        for nameStr, base_vector in zip(vector_names, system._base_vectors):
            setattr(system, nameStr.name, base_vector)
        for nameStr, base_scalar in zip(variable_names, system._base_scalars):
            setattr(system, nameStr.name, base_scalar)

    @staticmethod
    def _set_transformation_private_attributes(
        system, parent, origin, transformation, rotation_matrix
    ):
        lambda_cobm_to_cartesian = None
        location = origin._pos
        if not all(t == 0 for t in eye(3) - rotation_matrix):
            # rotated cartesian system
            lambda_transf_to_cartesian = CoordSys3D._compose_rotation_and_translation(
                rotation_matrix.T, location, parent)
            location = location._projections
            lambda_transf_from_cartesian = lambda x, y, z: rotation_matrix * Matrix([
                x - location[0],
                y - location[1],
                z - location[2]])
            template = _get_coordsys_template("cartesian")
            lambda_lame_coefficients = template.lame_coefficients
        elif isinstance(transformation, Str):
            # any pre-built system, even the cartesian Cartesian
            location = location._projections
            template = _get_coordsys_template(transformation.name)
            lambda_transf_to_cartesian = lambda x1, x2, x3: \
                tuple(c + l for c, l in zip(
                    template.transf_to_cartesian(x1, x2, x3), location))
            lambda_transf_from_cartesian = lambda x, y, z: \
                tuple(c - l for c, l in zip(
                    template.transf_from_cartesian(x, y, z), location))
            lambda_lame_coefficients = template.lame_coefficients
            lambda_cobm_to_cartesian = template.cobm_to_cartesian
        elif isinstance(transformation, Lambda):
            if not CoordSys3D._check_orthogonality(transformation):
                raise ValueError("The transformation equation does not "
                                 "create orthogonal coordinate system")
            lambda_transf_to_cartesian = transformation
            lambda_transf_from_cartesian = None
            lambda_lame_coefficients = CoordSys3D._calculate_lame_coeff(
                lambda_transf_to_cartesian)
        else:
            raise TypeError(
                f"type(transformation)={type(transformation).__name__}"
                " is not supported.")

        u = system._base_scalars
        h = lambda_lame_coefficients(*u)

        # Compute the change of basis matrix from this system to a
        # cartesian one.
        is_cartesian = all(t == S.One for t in h)
        is_curvilinear = False
        patterns = (BaseScalar, BaseScalarFuncOfTime)
        # detects coordinate systems like spherical/cylindrical,
        # where the Jacobian depends on the position
        has_base_scalars = lambda matrix: matrix.has(*patterns)

        if lambda_cobm_to_cartesian is not None:
            cob_matrix_to_cart = lambda_cobm_to_cartesian(*u)
            is_curvilinear = has_base_scalars(cob_matrix_to_cart)
        else:
            X = Matrix(lambda_transf_to_cartesian(*u))
            J = X.jacobian(u)
            is_curvilinear = has_base_scalars(J)

            if is_curvilinear:
                # normalized jacobian
                J = Matrix([(J.col(i) / h[i]).T for i in range(3)]).T

            cob_matrix_to_cart = J

        system._lame_coefficients = h
        system._lambda_transf_to_cartesian = lambda_transf_to_cartesian
        system._lambda_transf_from_cartesian = lambda_transf_from_cartesian
        system._cobm_to_cartesian = cob_matrix_to_cart.as_immutable()
        system._is_cartesian = is_cartesian
        system._is_curvilinear = is_curvilinear
        system._parent_rotation_matrix = rotation_matrix

    @classmethod
    def _process_input_args(
        cls, name, transformation, parent, location,
        rotation_matrix, vector_names, variable_names, time_symbol
    ):
        name = str(name)
        Vector = sympy.vector.Vector
        Point = sympy.vector.Point

        if time_symbol and (not isinstance(time_symbol, Symbol)):
            raise TypeError("`time_symbol` must be a Symbol or None.")
        if parent and (not isinstance(parent, CoordSys3D)):
            raise TypeError("`parent` should be a CoordSys3D or None.")
        if location and (not isinstance(location, Vector)):
            raise TypeError("`location` should be a Vector or None.")
        if rotation_matrix and (not isinstance(rotation_matrix, MatrixBase)):
            raise TypeError(
                "`rotation_matrix` should be a ImmutableMatrix instance.")
        if transformation and (
            (location is not None) or (rotation_matrix is not None)
        ):
            raise ValueError(
                "Specify either `transformation` or "
                "`location`/`rotation_matrix`.")

        # process transformation to the appropriate value
        if isinstance(transformation, (Tuple, tuple, list)):
            if isinstance(transformation[0], MatrixBase):
                # this happens when orient_new is called
                rotation_matrix = transformation[0]
                location = transformation[1]
            else:
                # this happens when user provides two tuples
                transformation = Lambda(
                    transformation[0], transformation[1])
        elif isinstance(transformation, Callable):
            x1, x2, x3 = symbols('x1 x2 x3', cls=Dummy)
            transformation = Lambda(
                (x1, x2, x3), transformation(x1, x2, x3))
        elif isinstance(transformation, str):
            transformation = Str(transformation)
        elif isinstance(transformation, (Str, Lambda)):
            pass
        elif transformation is None:
            transformation = Str("cartesian")
        else:
            raise TypeError(
                "Wrong type for `transformation`:"
                f" {type(transformation).__name__}")

        # If orientation information has been provided, store
        # the rotation matrix accordingly
        if rotation_matrix is None:
            rotation_matrix = eye(3)
        rotation_matrix = rotation_matrix.as_immutable()

        # If location information is not given, adjust the default
        # location as Vector.zero
        if location is None:
            location = Vector.zero
        else:
            # Check that location does not contain base scalars
            for x in location.free_symbols:
                if isinstance(x, (BaseScalar, BaseScalarFuncOfTime)):
                    raise ValueError(
                        "`location` should not contain BaseScalars.")

        if parent is not None:
            origin = parent.origin.locate_new(
                name + '.origin', location)
        else:
            origin = Point(name + '.origin')

        if isinstance(transformation, Str):
            template = _get_coordsys_template(transformation.name)
            if variable_names is None:
                variable_names = template.base_scalar_names
            if vector_names is None:
                vector_names = template.base_vector_names
        elif isinstance(transformation, Lambda):
            if variable_names is None:
                variable_names = ["x1", "x2", "x3"]
            if vector_names is None:
                vector_names = ["i", "j", "k"]
        else:
            if variable_names is None:
                variable_names = ["x", "y", "z"]
            if vector_names is None:
                vector_names = ["i", "j", "k"]

        variable_names = Tuple(*[Str(t) for t in variable_names])
        vector_names = Tuple(*[Str(t) for t in vector_names])

        # arguments of Basic must be instances of Basic too, otherwise
        # symbolic algorithm might fails (like substitutions)
        if parent is None:
            parent = sympify(False)
        if time_symbol is None:
            time_symbol = sympify(False)

        return (
            Str(name), transformation, parent, origin,
            rotation_matrix, vector_names, variable_names, time_symbol
        )

    def _sympystr(self, printer):
        return self.name

    def __iter__(self):
        return iter(self.base_vectors())

    def _eval_simplify(self, **kwargs):
        return self

    @staticmethod
    def _check_orthogonality(equations):
        """
        It checks if set of transformation equations create orthogonal
        curvilinear coordinate system.

        Parameters
        ==========

        equations : Lambda
            Lambda of transformation equations

        """

        x1, x2, x3 = symbols("x1, x2, x3", cls=Dummy)
        e1, e2, e3 = equations(x1, x2, x3)
        v1 = Matrix([diff(e1, x1), diff(e2, x1), diff(e3, x1)])
        v2 = Matrix([diff(e1, x2), diff(e2, x2), diff(e3, x2)])
        v3 = Matrix([diff(e1, x3), diff(e2, x3), diff(e3, x3)])

        if any(simplify(i[0] + i[1] + i[2]) == 0 for i in (v1, v2, v3)):
            return False
        else:
            if simplify(v1.dot(v2)) == 0 and simplify(v2.dot(v3)) == 0 \
                and simplify(v3.dot(v1)) == 0:
                return True
            else:
                return False

    def _calculate_inv_trans_equations(self):
        """
        Calculates the inverse transformation equations. Note that this
        procedure uses SymPy's ``solve``, hence it might take a while
        to compute the solutions.
        """
        x1, x2, x3 = symbols("x1, x2, x3", cls=Dummy, reals=True)
        x, y, z = symbols("x, y, z", cls=Dummy)

        transformation = self.transformation
        if isinstance(transformation, str):
            template = _get_coordsys_template(transformation)
            transformation = template.transf_to_cartesian

        e1, e2, e3 = transformation(x1, x2, x3)
        solved = solve([e1 - x, e2 - y, e3 - z], (x1, x2, x3), dict=True)[0]
        solved = solved[x1], solved[x2], solved[x3]
        self._lambda_transf_from_cartesian = lambda x1, x2, x3: \
            tuple(i.subs(list(zip((x, y, z), (x1, x2, x3)))) for i in solved)

    @staticmethod
    def _calculate_lame_coeff(equations):
        """
        Calculates Lame coefficients for given transformations equations.

        Parameters
        ==========

        equations : Lambda
            Lambda of transformation equations.

        """
        def func(x1, x2, x3):
            e1, e2, e3 = equations(x1, x2, x3)
            return (
                sqrt(diff(e1, x1)**2 + diff(e2, x1)**2 + diff(e3, x1)**2),
                sqrt(diff(e1, x2)**2 + diff(e2, x2)**2 + diff(e3, x2)**2),
                sqrt(diff(e1, x3)**2 + diff(e2, x3)**2 + diff(e3, x3)**2)
            )
        return func

    def _inverse_rotation_matrix(self):
        """
        Returns inverse rotation matrix.
        """
        return simplify(self._parent_rotation_matrix**-1)

    @classmethod
    def _rotation_trans_equations(cls, matrix, equations):
        """
        Returns the transformation equations obtained from rotation matrix.

        Parameters
        ==========

        matrix : Matrix
            Rotation matrix

        equations : tuple
            Transformation equations

        """
        return tuple(matrix * Matrix(equations))

    @property
    def origin(self):
        return self.args[3]

    def base_vectors(self):
        return self._base_vectors

    def base_scalars(self):
        return self._base_scalars

    def lame_coefficients(self):
        return self._lame_coefficients

    def transformation_to_parent(self):
        return self._lambda_transf_to_cartesian(*self.base_scalars())

    def transformation_from_parent(self):
        if self._lambda_transf_from_cartesian is None:
            self._calculate_inv_trans_equations()

        if self.parent is None:
            coords = symbols("x1, x2, x3", cls=Dummy)
        else:
            coords = self.parent.base_scalars()
        return self._lambda_transf_from_cartesian(*coords)

    def transformation_from_parent_function(self):
        if self._lambda_transf_from_cartesian is None:
            self._calculate_inv_trans_equations()
        return self._lambda_transf_from_cartesian

    def rotation_matrix(self, other):
        """
        Returns the direction cosine matrix(DCM), also known as the
        'rotation matrix' of this coordinate system with respect to
        another system.

        If v_a is a vector defined in system 'A' (in matrix format)
        and v_b is the same vector defined in system 'B', then
        v_a = A.rotation_matrix(B) * v_b.

        A SymPy Matrix is returned.

        Note that rotation_matrix only consider pure rotations between
        Cartesian systems. It doesn't account for changes in base vectors,
        like curvilinear to Cartesian. Use change_of_basis_matrix_from if
        those are important.

        Parameters
        ==========

        other : CoordSys3D
            The system which the DCM is generated to.

        See Also
        ========

        CoordSys3D.change_of_basis_matrix_from

        Examples
        ========

        >>> from sympy.vector import CoordSys3D
        >>> from sympy import symbols

        Let's consider two Cartesian systems connected by rotations:

        >>> q1 = symbols('q1')
        >>> N = CoordSys3D('N')
        >>> A = N.orient_new_axis('A', q1, N.i)

        The rotation matrix from A to N is:

        >>> R_fromA_toN = N.rotation_matrix(A)
        >>> R_fromA_toN
        Matrix([
        [1,       0,        0],
        [0, cos(q1), -sin(q1)],
        [0, sin(q1),  cos(q1)]])

        Because the two systems are Cartesians and connected by pure rotation,
        the rotation matrix is equal to the change-of-basis matrix:

        >>> R_fromA_toN == N.change_of_basis_matrix_from(A)
        True

        This is generally not true. Specifically if a non-Cartesian coordinate
        system (like curvilinear system, or a scaled system, or a
        reflected system, etc.) is in the path of two connected systems,
        then the two matrices are different, because the one computed by
        change_of_basis_matrix_from considers both the change in base vectors
        and the orientation between the systems, whereas the one computed by
        rotation_matrix only consider the orientation between Cartesian
        systems. For example:

        >>> B = A.create_new("B", transformation=lambda x,y,z: (2*x, z, y))
        >>> C = B.orient_new_axis("C", q1, B.i)
        >>> R_fromC_toN = N.rotation_matrix(C).simplify()
        >>> R_fromC_toN
        Matrix([
        [1,         0,          0],
        [0, cos(2*q1), -sin(2*q1)],
        [0, sin(2*q1),  cos(2*q1)]])
        >>> T_fromC_toN = N.change_of_basis_matrix_from(C).simplify()
        >>> T_fromC_toN
        Matrix([
        [2, 0, 0],
        [0, 0, 1],
        [0, 1, 0]])

        Hence, the use of change_of_basis_matrix_from is recommended.

        """
        from sympy.vector.functions import _path
        if not isinstance(other, CoordSys3D):
            raise TypeError(str(other) +
                            " is not a CoordSys3D")
        # Handle special cases
        if other == self:
            return eye(3)
        elif other == self.parent:
            return self._parent_rotation_matrix
        elif other.parent == self:
            return other._parent_rotation_matrix.T
        # Else, use tree to calculate position
        rootindex, path = _path(self, other)
        result = eye(3)
        for i in range(rootindex):
            result *= path[i]._parent_rotation_matrix
        for i in range(rootindex + 1, len(path)):
            result *= path[i]._parent_rotation_matrix.T
        return result

    def change_of_basis_matrix_from(self, other):
        """
        Returns the change-of-basis matrix from the ``other`` coordinate system
        to this coordinate system. This matrix transforms the components of a
        vector defined in ``other`` to componenets of a vector defined in
        this system. The columns of this matrix represent the base vectors of
        the ``other`` system in terms of base vectors of this
        coordinate system.

        If v_a is a vector defined in system 'A' (in matrix format)
        and v_b is the same vector defined in system 'B', then
        v_a = A.change_of_basis_matrix_from(B) * v_b.

        A SymPy Matrix is returned.

        Parameters
        ==========

        other : CoordSys3D
            The system from which the change-of-basis matrix is generated.

        See Also
        ========

        CoordSys3D.rotation_matrix

        Examples
        ========

        >>> from sympy.vector import CoordSys3D, express
        >>> from sympy import symbols, Matrix

        Let's consider two Cartesian systems connected by rotations:

        >>> q1 = symbols('q1')
        >>> N = CoordSys3D('N')
        >>> A = N.orient_new_axis('A', q1, N.i)

        The change-of-basis matrix (or transformation matrix) from A to N is:

        >>> T_fromA_toN = N.change_of_basis_matrix_from(A)
        >>> T_fromA_toN
        Matrix([
        [1,       0,        0],
        [0, cos(q1), -sin(q1)],
        [0, sin(q1),  cos(q1)]])

        The change-of-basis matrix allows to express vectors defined
        in one system into a different system. In fact, the change-of-basis
        matrix is internally used by the ``express`` function. The following
        example shows a direct comparison between working with matrices
        and the vector module:

        >>> vA_matrix = Matrix([1, 2, 3])
        >>> vN_matrix = T_fromA_toN * vA_matrix
        >>> vN_matrix
        Matrix([
        [                     1],
        [-3*sin(q1) + 2*cos(q1)],
        [ 2*sin(q1) + 3*cos(q1)]])
        >>> vA = A.i + 2 * A.j + 3 * A.k
        >>> vN = express(vA, N)
        >>> vN
        N.i + (-3*sin(q1) + 2*cos(q1))*N.j + (2*sin(q1) + 3*cos(q1))*N.k

        Change-of-basis matrix from spherical to Cartesian coordinates:

        >>> Cart = CoordSys3D("Cart")
        >>> S = Cart.create_new("S", transformation="spherical")
        >>> C = Cart.create_new("C", transformation="cylindrical")
        >>> Cart.change_of_basis_matrix_from(S)
        Matrix([
        [sin(S.theta)*cos(S.phi), cos(S.phi)*cos(S.theta), -sin(S.phi)],
        [sin(S.phi)*sin(S.theta), sin(S.phi)*cos(S.theta),  cos(S.phi)],
        [           cos(S.theta),           -sin(S.theta),           0]])

        Change-of-basis matrix from spherical to cylindrical coordinates
        (note that the azimuthal angle of S and C are the same):

        >>> r_s, theta_s, phi_s = S.base_scalars()
        >>> r_c, theta_c, z_c = C.base_scalars()
        >>> C.change_of_basis_matrix_from(S).subs(phi_s, theta_c).simplify()
        Matrix([
        [sin(S.theta),  cos(S.theta), 0],
        [           0,             0, 1],
        [cos(S.theta), -sin(S.theta), 0]])

        """

        from sympy.vector.functions import _path
        if not isinstance(other, CoordSys3D):
            raise TypeError(str(other) + " is not a CoordSys3D")
        if self == other:
            return eye(3)

        # Else, use tree to calculate position
        rootindex, path = _path(self, other)
        result = eye(3)
        for i in range(rootindex):
            if path[i]._is_cartesian or path[i]._is_curvilinear:
                result *= path[i]._cobm_to_cartesian.T
            else:
                result *= path[i]._cobm_to_cartesian.inv()
        for i in range(rootindex + 1, len(path)):
            result *= path[i]._cobm_to_cartesian

        # attempts to cancel out opposite rotations
        result = TR5(result)
        return result

    @cacheit
    def position_wrt(self, other):
        """
        Returns the position vector of the origin of this coordinate
        system with respect to another Point/CoordSys3D.

        Parameters
        ==========

        other : Point/CoordSys3D
            If other is a Point, the position of this system's origin
            wrt it is returned. If its an instance of CoordSyRect,
            the position wrt its origin is returned.

        Examples
        ========

        >>> from sympy.vector import CoordSys3D
        >>> N = CoordSys3D('N')
        >>> N1 = N.locate_new('N1', 10 * N.i)
        >>> N.position_wrt(N1)
        (-10)*N.i

        """
        return self.origin.position_wrt(other)

    def scalar_map(self, other):
        """
        Returns a dictionary which expresses the coordinate variables
        (base scalars) of this frame in terms of the variables of
        otherframe.

        Parameters
        ==========

        otherframe : CoordSys3D
            The other system to map the variables to.

        Examples
        ========

        >>> from sympy.vector import CoordSys3D
        >>> from sympy import Symbol
        >>> A = CoordSys3D('A')
        >>> q = Symbol('q')
        >>> B = A.orient_new_axis('B', q, A.k)
        >>> A.scalar_map(B)
        {A.x: B.x*cos(q) - B.y*sin(q), A.y: B.x*sin(q) + B.y*cos(q), A.z: B.z}

        """

        from sympy.vector.functions import _path

        root_idx, systems = _path(other, self)
        fx, fy, fz = self.base_scalars()
        current = {fx: fx, fy: fy, fz: fz}

        for system in systems[:root_idx]:
            px, py, pz = system.transformation_to_parent()
            parent = system.parent
            px, py, pz = px.subs(current), py.subs(current), pz.subs(current)
            pxs, pys, pzs = parent.base_scalars()
            current = {pxs: px, pys: py, pzs: pz}

        for system in systems[root_idx + 1:]:
            xs, ys, zs = system.transformation_from_parent()
            xs, ys, zs = [t.subs(current) for t in [xs, ys, zs]]
            sx, sy, sz = system.base_scalars()
            current = {sx: xs, sy: ys, sz: zs}

        def post_process(expr):
            # attempts to cancel out opposite rotations
            def pattern(t):
                return (
                    t is not None
                    and t.is_Mul
                    and any(isinstance(a, TrigonometricFunction) for a in t.args)
                    and any(a.is_Add and a.has(TrigonometricFunction) for a in t.args)
                )

            if not expr.find(pattern):
                return expr
            return TR5(factor_terms(expr.expand()))

        current = {k: post_process(v) for k, v in current.items()}
        return current

    def locate_new(self, name, position, vector_names=None,
                   variable_names=None):
        """
        Returns a CoordSys3D with its origin located at the given
        position wrt this coordinate system's origin.

        Parameters
        ==========

        name : str
            The name of the new CoordSys3D instance.

        position : Vector
            The position vector of the new system's origin wrt this
            one.

        vector_names, variable_names : iterable(optional)
            Iterables of 3 strings each, with custom names for base
            vectors and base scalars of the new system respectively.
            Used for simple str printing.

        Examples
        ========

        >>> from sympy.vector import CoordSys3D
        >>> A = CoordSys3D('A')
        >>> B = A.locate_new('B', 10 * A.i)
        >>> B.origin.position_wrt(A.origin)
        10*A.i

        """
        if variable_names is None:
            variable_names = self.variable_names
        if vector_names is None:
            vector_names = self.vector_names

        return CoordSys3D(name, location=position,
                          vector_names=vector_names,
                          variable_names=variable_names,
                          parent=self)

    def orient_new(self, name, orienters, location=None,
                   vector_names=None, variable_names=None):
        """
        Creates a new CoordSys3D oriented in the user-specified way
        with respect to this system.

        Please refer to the documentation of the orienter classes
        for more information about the orientation procedure.

        Parameters
        ==========

        name : str
            The name of the new CoordSys3D instance.

        orienters : iterable/Orienter
            An Orienter or an iterable of Orienters for orienting the
            new coordinate system.
            If an Orienter is provided, it is applied to get the new
            system.
            If an iterable is provided, the orienters will be applied
            in the order in which they appear in the iterable.

        location : Vector(optional)
            The location of the new coordinate system's origin wrt this
            system's origin. If not specified, the origins are taken to
            be coincident.

        vector_names, variable_names : iterable(optional)
            Iterables of 3 strings each, with custom names for base
            vectors and base scalars of the new system respectively.
            Used for simple str printing.

        Examples
        ========

        >>> from sympy.vector import CoordSys3D
        >>> from sympy import symbols
        >>> q0, q1, q2, q3 = symbols('q0 q1 q2 q3')
        >>> N = CoordSys3D('N')

        Using an AxisOrienter

        >>> from sympy.vector import AxisOrienter
        >>> axis_orienter = AxisOrienter(q1, N.i + 2 * N.j)
        >>> A = N.orient_new('A', (axis_orienter, ))

        Using a BodyOrienter

        >>> from sympy.vector import BodyOrienter
        >>> body_orienter = BodyOrienter(q1, q2, q3, '123')
        >>> B = N.orient_new('B', (body_orienter, ))

        Using a SpaceOrienter

        >>> from sympy.vector import SpaceOrienter
        >>> space_orienter = SpaceOrienter(q1, q2, q3, '312')
        >>> C = N.orient_new('C', (space_orienter, ))

        Using a QuaternionOrienter

        >>> from sympy.vector import QuaternionOrienter
        >>> q_orienter = QuaternionOrienter(q0, q1, q2, q3)
        >>> D = N.orient_new('D', (q_orienter, ))
        """
        if variable_names is None:
            variable_names = self.variable_names
        if vector_names is None:
            vector_names = self.vector_names

        if isinstance(orienters, Orienter):
            if isinstance(orienters, AxisOrienter):
                final_matrix = orienters.rotation_matrix(self)
            else:
                final_matrix = orienters.rotation_matrix()
            # TODO: trigsimp is needed here so that the matrix becomes
            # canonical (scalar_map also calls trigsimp; without this, you can
            # end up with the same CoordinateSystem that compares differently
            # due to a differently formatted matrix). However, this is
            # probably not so good for performance.
            final_matrix = trigsimp(final_matrix)
        else:
            final_matrix = Matrix(eye(3))
            for orienter in orienters:
                if isinstance(orienter, AxisOrienter):
                    final_matrix *= orienter.rotation_matrix(self)
                else:
                    final_matrix *= orienter.rotation_matrix()

        return CoordSys3D(name, rotation_matrix=final_matrix,
                          vector_names=vector_names,
                          variable_names=variable_names,
                          location=location,
                          parent=self)

    def orient_new_axis(self, name, angle, axis, location=None,
                        vector_names=None, variable_names=None):
        """
        Axis rotation is a rotation about an arbitrary axis by
        some angle. The angle is supplied as a SymPy expr scalar, and
        the axis is supplied as a Vector.

        Parameters
        ==========

        name : string
            The name of the new coordinate system

        angle : Expr
            The angle by which the new system is to be rotated

        axis : Vector
            The axis around which the rotation has to be performed

        location : Vector(optional)
            The location of the new coordinate system's origin wrt this
            system's origin. If not specified, the origins are taken to
            be coincident.

        vector_names, variable_names : iterable(optional)
            Iterables of 3 strings each, with custom names for base
            vectors and base scalars of the new system respectively.
            Used for simple str printing.

        Examples
        ========

        >>> from sympy.vector import CoordSys3D
        >>> from sympy import symbols
        >>> q1 = symbols('q1')
        >>> N = CoordSys3D('N')
        >>> B = N.orient_new_axis('B', q1, N.i + 2 * N.j)

        """
        if variable_names is None:
            variable_names = self.variable_names
        if vector_names is None:
            vector_names = self.vector_names

        orienter = AxisOrienter(angle, axis)
        return self.orient_new(name, orienter,
                               location=location,
                               vector_names=vector_names,
                               variable_names=variable_names)

    def orient_new_body(self, name, angle1, angle2, angle3,
                        rotation_order, location=None,
                        vector_names=None, variable_names=None):
        """
        Body orientation takes this coordinate system through three
        successive simple rotations.

        Body fixed rotations include both Euler Angles and
        Tait-Bryan Angles, see https://en.wikipedia.org/wiki/Euler_angles.

        Parameters
        ==========

        name : string
            The name of the new coordinate system

        angle1, angle2, angle3 : Expr
            Three successive angles to rotate the coordinate system by

        rotation_order : string
            String defining the order of axes for rotation

        location : Vector(optional)
            The location of the new coordinate system's origin wrt this
            system's origin. If not specified, the origins are taken to
            be coincident.

        vector_names, variable_names : iterable(optional)
            Iterables of 3 strings each, with custom names for base
            vectors and base scalars of the new system respectively.
            Used for simple str printing.

        Examples
        ========

        >>> from sympy.vector import CoordSys3D
        >>> from sympy import symbols
        >>> q1, q2, q3 = symbols('q1 q2 q3')
        >>> N = CoordSys3D('N')

        A 'Body' fixed rotation is described by three angles and
        three body-fixed rotation axes. To orient a coordinate system D
        with respect to N, each sequential rotation is always about
        the orthogonal unit vectors fixed to D. For example, a '123'
        rotation will specify rotations about N.i, then D.j, then
        D.k. (Initially, D.i is same as N.i)
        Therefore,

        >>> D = N.orient_new_body('D', q1, q2, q3, '123')

        is same as

        >>> D = N.orient_new_axis('D', q1, N.i)
        >>> D = D.orient_new_axis('D', q2, D.j)
        >>> D = D.orient_new_axis('D', q3, D.k)

        Acceptable rotation orders are of length 3, expressed in XYZ or
        123, and cannot have a rotation about about an axis twice in a row.

        >>> B = N.orient_new_body('B', q1, q2, q3, '123')
        >>> B = N.orient_new_body('B', q1, q2, 0, 'ZXZ')
        >>> B = N.orient_new_body('B', 0, 0, 0, 'XYX')

        """

        orienter = BodyOrienter(angle1, angle2, angle3, rotation_order)
        return self.orient_new(name, orienter,
                               location=location,
                               vector_names=vector_names,
                               variable_names=variable_names)

    def orient_new_space(self, name, angle1, angle2, angle3,
                         rotation_order, location=None,
                         vector_names=None, variable_names=None):
        """
        Space rotation is similar to Body rotation, but the rotations
        are applied in the opposite order.

        Parameters
        ==========

        name : string
            The name of the new coordinate system

        angle1, angle2, angle3 : Expr
            Three successive angles to rotate the coordinate system by

        rotation_order : string
            String defining the order of axes for rotation

        location : Vector(optional)
            The location of the new coordinate system's origin wrt this
            system's origin. If not specified, the origins are taken to
            be coincident.

        vector_names, variable_names : iterable(optional)
            Iterables of 3 strings each, with custom names for base
            vectors and base scalars of the new system respectively.
            Used for simple str printing.

        See Also
        ========

        CoordSys3D.orient_new_body : method to orient via Euler
            angles

        Examples
        ========

        >>> from sympy.vector import CoordSys3D
        >>> from sympy import symbols
        >>> q1, q2, q3 = symbols('q1 q2 q3')
        >>> N = CoordSys3D('N')

        To orient a coordinate system D with respect to N, each
        sequential rotation is always about N's orthogonal unit vectors.
        For example, a '123' rotation will specify rotations about
        N.i, then N.j, then N.k.
        Therefore,

        >>> D = N.orient_new_space('D', q1, q2, q3, '312')

        is same as

        >>> B = N.orient_new_axis('B', q1, N.i)
        >>> C = B.orient_new_axis('C', q2, N.j)
        >>> D = C.orient_new_axis('D', q3, N.k)

        """

        orienter = SpaceOrienter(angle1, angle2, angle3, rotation_order)
        return self.orient_new(name, orienter,
                               location=location,
                               vector_names=vector_names,
                               variable_names=variable_names)

    def orient_new_quaternion(self, name, q0, q1, q2, q3, location=None,
                              vector_names=None, variable_names=None):
        """
        Quaternion orientation orients the new CoordSys3D with
        Quaternions, defined as a finite rotation about lambda, a unit
        vector, by some amount theta.

        This orientation is described by four parameters:

        q0 = cos(theta/2)

        q1 = lambda_x sin(theta/2)

        q2 = lambda_y sin(theta/2)

        q3 = lambda_z sin(theta/2)

        Quaternion does not take in a rotation order.

        Parameters
        ==========

        name : string
            The name of the new coordinate system

        q0, q1, q2, q3 : Expr
            The quaternions to rotate the coordinate system by

        location : Vector(optional)
            The location of the new coordinate system's origin wrt this
            system's origin. If not specified, the origins are taken to
            be coincident.

        vector_names, variable_names : iterable(optional)
            Iterables of 3 strings each, with custom names for base
            vectors and base scalars of the new system respectively.
            Used for simple str printing.

        Examples
        ========

        >>> from sympy.vector import CoordSys3D
        >>> from sympy import symbols
        >>> q0, q1, q2, q3 = symbols('q0 q1 q2 q3')
        >>> N = CoordSys3D('N')
        >>> B = N.orient_new_quaternion('B', q0, q1, q2, q3)

        """

        orienter = QuaternionOrienter(q0, q1, q2, q3)
        return self.orient_new(name, orienter,
                               location=location,
                               vector_names=vector_names,
                               variable_names=variable_names)

    def create_new(self, name, transformation, variable_names=None, vector_names=None):
        """
        Returns a CoordSys3D which is connected to self by transformation.

        Parameters
        ==========

        name : str
            The name of the new CoordSys3D instance.

        transformation : Lambda, Tuple, str
            Transformation defined by transformation equations or chosen
            from predefined ones.

        vector_names, variable_names : iterable(optional)
            Iterables of 3 strings each, with custom names for base
            vectors and base scalars of the new system respectively.
            Used for simple str printing.

        Examples
        ========

        >>> from sympy.vector import CoordSys3D
        >>> a = CoordSys3D('a')
        >>> b = a.create_new('b', transformation='spherical')
        >>> b.transformation_to_parent()
        (b.r*sin(b.theta)*cos(b.phi), b.r*sin(b.phi)*sin(b.theta), b.r*cos(b.theta))
        >>> b.transformation_from_parent()
        (sqrt(a.x**2 + a.y**2 + a.z**2), acos(a.z/sqrt(a.x**2 + a.y**2 + a.z**2)), atan2(a.y, a.x))

        """
        return CoordSys3D(name, parent=self, transformation=transformation,
                          variable_names=variable_names, vector_names=vector_names)

    def __init__(self, name, location=None, rotation_matrix=None,
                 parent=None, vector_names=None, variable_names=None,
                 latex_vects=None, pretty_vects=None, latex_scalars=None,
                 pretty_scalars=None, transformation=None, time_symbol=None):
        # Dummy initializer for setting docstring
        pass

    __init__.__doc__ = __new__.__doc__

    @staticmethod
    def _compose_rotation_and_translation(rot, translation, parent):
        r = lambda x, y, z: CoordSys3D._rotation_trans_equations(rot, (x, y, z))
        if parent is None:
            return r

        dx, dy, dz = [translation.dot(i) for i in parent.base_vectors()]
        t = lambda x, y, z: (x + dx, y + dy, z + dz)
        return lambda x, y, z: t(*r(x, y, z))


def _check_strings(arg_name, arg):
    errorstr = arg_name + " must be an iterable of 3 string-types"
    if len(arg) != 3:
        raise ValueError(errorstr)
    for s in arg:
        if not isinstance(s, str):
            raise TypeError(errorstr)


# Delayed import to avoid cyclic import problems:
from sympy.vector.vector import BaseVector
