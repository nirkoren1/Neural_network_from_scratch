package Matrix;
import java.util.Random;

public class Matrix implements IMatrix {
    private int rows;
    private int columns;
    private Double[][] values;
    private Random random = new Random();

    public Matrix(int rows, int columns) {
        this.rows = rows;
        this.columns = columns;
        this.values = new Double[rows][columns];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < columns; j++) {
                this.values[i][j] = -1 + (1 - -1) * random.nextDouble();
            }
        }
    }

    public Matrix(int rows, int columns, Double[][] values) {
        this.rows = rows;
        this.columns = columns;
        this.values = new Double[rows][columns];
        for (int i = 0; i < rows; i++) {
            System.arraycopy(values[i], 0, this.values[i], 0, columns);
        }
    }
    @Override
    public int getRows() {
        return this.rows;
    }

    @Override
    public int getColumns() {
        return this.columns;
    }

    @Override
    public Double[][] getValues() {
        return this.values;
    }

    public static Matrix trancePose(Matrix first) {
        Double[][] newValues = new Double[first.getColumns()][first.getRows()];
        for (int i = 0; i < first.getRows(); i++) {
            for (int j = 0; j < first.getColumns(); j++) {
                newValues[j][i] = first.getValues()[i][j];
            }
        }
        return new Matrix(first.getColumns(), first.getRows(), newValues);
    }
    public static Matrix add(Matrix first, Matrix second) {
        if (first.getColumns() != second.getColumns() || first.getRows() != second.getRows()) {
            System.out.println("matrix add doesn't match for shape (" + first.getRows() + ", " + first.getColumns() + ")  (" +
                                second.getRows() + ", " + second.getColumns() + ")");
            return null;
        }
        Double[][] newValues = first.values;
        for (int i = 0; i < first.getRows(); i++) {
            for (int j = 0; j < first.getColumns(); j++) {
                newValues[i][j] += second.values[i][j];
            }
        }
        return new Matrix(first.getRows(), first.getColumns(), newValues);
    }

    public static Matrix dot(Matrix first, Matrix second) {
        if (first.getColumns() != second.getRows()) {
            System.out.println("matrix dot doesn't match for shape (" + first.getRows() + ", " + first.getColumns() + ")  (" +
                    second.getRows() + ", " + second.getColumns() + ")");
            return null;
        }
        Double[][] newValues = new Double[first.getRows()][second.getColumns()];
        for (int i = 0; i < first.getRows(); i++) {
            for (int j = 0; j < second.getColumns(); j++) {
                double val = 0;
                for (int k = 0; k < first.getColumns(); k++) {
                    val += first.getValues()[i][k] * second.getValues()[k][j];
                }
                newValues[i][j] = val;
            }
        }
        return new Matrix(first.getRows(), second.getColumns(), newValues);
    }

    public static Matrix multiply(Matrix first, double scalar) {
        Double[][] newValues = first.getValues().clone();
        for (int i = 0; i < first.getRows(); i++) {
            for (int j = 0; j < first.getColumns(); j++) {
                newValues[i][j] *= scalar;
            }
        }
        return new Matrix(first.getRows(), first.getColumns(), newValues);
    }

    public static Matrix getEMatrix(int rows, int columns, int i, int j) {
        Matrix out = new Matrix(rows, columns);
        for (int k = 0; k < rows; k++) {
            for (int l = 0; l < columns; l++) {
                if (i == k && j == l) {
                    out.getValues()[k][l] = 1.0;
                } else {
                    out.getValues()[k][l] = 0.0;
                }
            }
        }
        return out;
    }

    public static Matrix diagonalizeVector(Double[] vector) {
        Matrix out = new Matrix(vector.length, vector.length);
        for (int i = 0; i < vector.length; i++) {
            for (int j = 0; j < vector.length; j++) {
                if (i == j) {
                    out.getValues()[i][j] = vector[i];
                } else {
                    out.getValues()[i][j] = 0.0;
                }
            }
        }
        return out;
    }

    @Override
    public String toString() {
        StringBuilder out = new StringBuilder();
        for (int i = 0; i < this.rows; i++) {
            for (int j = 0; j < this.columns; j++) {
                out.append(this.values[i][j]);
                out.append("  ");
            }
            out.append("\n");
        }
        return out.toString();
    }

    public Matrix copy() {
        return new Matrix(this.getRows(), this.getColumns(), this.getValues());
    }

    public static void main(String[] args) {
        Matrix matrix = new Matrix(2, 3);
        System.out.println(matrix);
        matrix = Matrix.trancePose(matrix);
        System.out.println(matrix);
    }
}
