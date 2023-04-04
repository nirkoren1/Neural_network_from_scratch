package Matrix;

public interface IMatrix {
    public int getRows();
    public int getColumns();

    public static Matrix multiply(Matrix first, double scalar) {
        return null;
    }

    public static Matrix dot(Matrix first, Matrix second) {
        return null;
    }

    public static Matrix add(Matrix first, Matrix second) {
        return null;
    }

    public Double[][] getValues();

    public static Matrix trancePose() {
        return null;
    }

    public static Matrix diagonalizeVector(Double[] vector) {
        return null;
    }

    public static Matrix dotElementWise(Matrix first, Matrix second) {return null;}

    public static Matrix divideElementWise(Matrix first, Matrix second) {return null;}

    public static Matrix identity(int rows, int cols) {return null;}

    public void print();
}
