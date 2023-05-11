
import operator

ops = {"+": operator.add,
       "-": operator.sub,
       "*": operator.mul,
       "/": operator.truediv}


class Matrix:

    def __init__(self, arg) -> None:
        if (arg is None):
            print("Nothing got in")
            return
        self.data = None
        self.shape = None
        if isinstance(arg, list):
            if all(len(sublist) == len(arg[0]) for sublist in arg):
                self.data = arg
                self.shape = (len(self.data), len(self.data[0]))
            else:
                print("Not all sublists have the same length")
            return
        elif isinstance(arg, tuple):
            self.data = [[0.0 for j in range(arg[0])] for i in range(arg[1])]
            self.shape = arg
            return

    def __str__(self):
        return f"Matrix({self.data})"

    def __repr__(self):
        return f"Matrix({self.data})"

    def __add__(self, other):
        if not isinstance(other, Matrix):
            raise TypeError("only matrix can add to each other")
        if (self.shape != other.shape):
            raise ValueError("only matrix of same shape is allowed")

        tmp = []
        # print(self.data[0] + other.data[0], self.shape[0])
        for i in range(0, self.shape[0]):
            tmp.append([a + b for a, b in zip(self.data[i], other.data[i])])
        return Matrix(tmp)

    def __radd__(self, other):
        if not isinstance(other, Matrix):
            raise TypeError("only matrix can add to each other")
        if (self.shape != other.shape):
            raise ValueError("only matrix of same shape is allowed")
        return other + self

    def __sub__(self, other):
        if not isinstance(other, Matrix):
            raise TypeError("only matrix can subtract from each other")
        if self.shape != other.shape:
            raise ValueError("only matrix of same shape is allowed")

        tmp = []
        for i in range(0, self.shape[0]):
            # print(self.data[i], other.data[i])
            tmp.append([a - b for a, b in zip(self.data[i], other.data[i])])

        return Matrix(tmp)

    def __rsub__(self, other):
        if not isinstance(other, Matrix):
            raise TypeError("only matrix can subtract from each other")
        if self.shape != other.shape:
            raise ValueError("only matrix of same shape is allowed")
        return other - self

    def __truediv__(self, var):
        if isinstance(var, Vector):
            raise NotImplementedError(
                "Division of a Matrix by a Matrix is not implemented here.")
        if not any([isinstance(var, t) for t in [float, int, complex]]):
            raise ValueError("division only accepts scalar. (real number)")
        if var == 0:
            raise ValueError("Division of 0 not allowed.")
        tmp = []
        for i in range(0, self.shape[0]):
            # print(self.data[i], other.data[i])
            tmp.append([a / var for a in self.data[i]])
        return Matrix(tmp)

    def __rtruediv__(self, var):
        raise NotImplementedError("rtruediv not implemented")

    def __mul__(self, var):
        # print(type(var))
        if any(isinstance(var, scalar_type) for scalar_type in [int, float, complex]):
            result = [[self.data[i][j] *
                       var for j in range(self.shape[1])] for i in range(self.shape[0])]
            return Matrix(result)
        elif isinstance(var, Vector):
            if self.shape[1] != var.shape[0]:
                raise ValueError(
                    "Matrices cannot be multiplied, dimensions don't match.")
            result = [[sum([self.data[i][k] * var.data[k][j] for k in range(self.shape[1])])
                       for j in range(var.shape[1])] for i in range(self.shape[0])]
            return Vector(result)
        elif isinstance(var, Matrix):
            # print("b", self.shape[1], var.shape[0])
            if self.shape[1] != var.shape[0]:
                raise ValueError(
                    "Matrices cannot be multiplied, dimensions don't match.")
            result = [[sum([self.data[i][k] * var.data[k][j] for k in range(self.shape[1])])
                       for j in range(var.shape[1])] for i in range(self.shape[0])]
            return Matrix(result)
        else:
            raise TypeError("Invalid type of input value.")

    def __rmul__(self, x):
        return self * x

    # ref: https://stackoverflow.com/questions/21444338/transpose-nested-list-in-python

    def T(self):
        return Matrix(list(map(list, zip(*self.data))))


class Vector(Matrix):

    def __init__(self, data):
        # a list of a list of floats: Vector([[0.0, 1.0, 2.0, 3.0]]),
        # • a list of lists of single float: Vector([[0.0], [1.0], [2.0], [3.0]]),
        # • a size: Vector(3) -> the vector will have values = [[0.0], [1.0], [2.0]],
        # • a range: Vector((10,16)) -> the vector will have values = [[10.0], [11.0],
        #            [12.0], [13.0], [14.0], [15.0]]. in Vector((a,b)), if a > b, you must display accurate error message
        ##
        if (isinstance(data, int)):
            if (data < 0):
                raise ValueError(
                    "Vector must be initialized with appropriate data (int, negative)")
            self.data = []
            for i in range(data):
                self.data.append([float(i)])
        elif (isinstance(data, tuple)):
            if not (len(data) == 2):
                raise ValueError(
                    "Vector must be initialized with appropriate data (tuple, length)")
            if not (isinstance(data[0], int) and isinstance(data[1], int)):
                raise ValueError(
                    "Vector must be initialized with appropriate data (tuple, data type)")
            if not (data[0] < data[1]):
                raise ValueError(
                    "Vector must be initialized with appropriate data (tuple, range)")
            self.data = []
            for i in range(data[0], data[1]):
                self.data.append([float(i)])
        # elif not (any(isinstance(i, list) for i in data) and isinstance(data, list)):
            # raise TypeError("vector must be initialized with appropriate data")
        else:
            # for list_inside in data:
            #     if isinstance(list_inside, list):
            #         for j in list_inside:
            #             if not (isinstance(j, float)):
            #                 raise TypeError(
            #                     "The element must be float type")
            #     else:
            #         if not (isinstance(list_inside, float)):
            #             raise TypeError("The element must be float type")

            self.data = data

        if len(self.data) == 1:
            # print(len(data[0]))
            self.shape = (1, len(self.data[0]))
        else:
            # print(len(data))
            self.shape = (len(self.data), 1)

    def __str__(self):
        return f"Vector({self.data})"

    def __repr__(self):
        return f"Vector({self.data})"

    def dot(self, other):
        if not isinstance(other, Vector):
            raise TypeError(
                "unsupported operand type(s) for +: '{}' and '{}'".format(type(self), type(other)))
        if self.shape[1] != other.shape[0]:
            raise TypeError(
                "Invalid input: dot product requires a Vector of compatible shape.")
        result = 0.0
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                result += self.data[i][j] * other.data[j][i]
        return result

    def _T_row_to_col(self):
        tmp = []
        for i in range(0, max(self.shape)):
            tmp.append([self.data[0][i]])
        return Vector(tmp)

    def _T_col_to_row(self):
        tmp = []
        for i in range(0, max(self.shape)):
            tmp.append(self.data[i][0])
        return Vector([tmp])

    def T(self):
        dimension_check = self.shape.index(max(self.shape))
        if (self.shape == (1, 1)):
            return self
        if dimension_check == 1:
            return self._T_row_to_col()
        else:
            return self._T_col_to_row()

    def __add__(self, other):
        if not isinstance(other, Vector):
            raise TypeError("only vector can add to each other")
        if (self.shape != other.shape):
            raise ValueError("only vector of same shape is allowed")
        dimension_check = self.shape.index(max(self.shape))
        tmp = []
        if dimension_check == 1:
            for i in range(0, max(self.shape)):
                tmp.append(self.data[0][i] + other.data[0][i])
            return Vector([tmp])
        else:
            for i in range(0, max(self.shape)):
                tmp.append([self.data[i][0] + other.data[i][0]])
            return Vector(tmp)

    def __radd__(self, other):
        return other + self

    def __sub__(self, other):
        if not isinstance(other, Vector):
            raise TypeError("only vector can add to each other")
        if (self.shape != other.shape):
            raise ValueError("only vector of same shape is allowed")
        dimension_check = self.shape.index(max(self.shape))
        tmp = []
        if dimension_check == 1:
            for i in range(0, max(self.shape)):
                tmp.append(self.data[0][i] - other.data[0][i])
            return Vector([tmp])
        else:
            for i in range(0, max(self.shape)):
                tmp.append([self.data[i][0] - other.data[i][0]])
            return Vector(tmp)

    def __rsub__(self, other):
        return other - self

    def __row_loop(self, var, operator):
        tmp = []
        for i in range(0, max(self.shape)):
            tmp.append(ops[operator](self.data[0][i], var))
        return Vector([tmp])

    def __col_loop(self, var, operator):
        tmp = []
        for i in range(0, max(self.shape)):
            tmp.append([ops[operator](self.data[i][0], var)])
        return Vector(tmp)

    def col_loop(self, other, operator):
        tmp = []
        for i in range(0, max(self.shape)):
            tmp.append([ops[operator](self.data[0][i], other.data[0][i])])
        return Vector([tmp])

    def __truediv__(self, var):
        if isinstance(var, Vector):
            raise NotImplementedError(
                "Division of a Vector by a Vector is not implemented here.")
        if not any([isinstance(var, t) for t in [float, int, complex]]):
            raise ValueError("division only accepts scalar. (real number)")
        dimension_check = self.shape.index(max(self.shape))
        if dimension_check == 1:
            return self.__row_loop(var, "/")
        else:
            return self.__col_loop(var, "/")

    def __rtruediv__(self, var):
        raise NotImplementedError(
            "Division of a scalar by a Vector is not implemented here.")

    def __mul__(self, other):
        if any(isinstance(other, scalar_type) for scalar_type in [int, float, complex]):
            result = [
                [self.data[i][j] * other for j in range(self.shape[1])] for i in range(self.shape[0])]
            return Vector(result)
        elif isinstance(other, Vector):
            if self.shape[1] != other.shape[0]:
                raise ValueError(
                    "Vectors cannot be multiplied, dimensions don't match.")
            result = [
                [self.data[i][j] * other for j in range(self.shape[1])] for i in range(self.shape[0])]
            return Vector(result)
        elif isinstance(other, Matrix):
            if self.shape[1] != other.shape[0]:
                raise ValueError(
                    "Matrices cannot be multiplied, dimensions don't match.")
            result = [[sum([self.data[i][k] * other.data[k][j] for k in range(self.shape[1])])
                       for j in range(other.shape[1])] for i in range(self.shape[0])]
            return Matrix(result)
        else:
            raise TypeError("Invalid type of input value.")
