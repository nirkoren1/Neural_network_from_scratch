package Matrix;
import java.io.Serializable;
import java.util.Random;

public class Matrix implements IMatrix, Serializable {
    private int rows;
    private int columns;
    private Double[][] values;
    private Random random = new Random(1);

    public Matrix(int rows, int columns) {
        this.rows = rows;
        this.columns = columns;
        this.values = new Double[rows][columns];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < columns; j++) {
//                this.values[i][j] = -1 + (1 - -1) * random.nextDouble();
                this.values[i][j] = random.nextGaussian();
            }
        }
    }

    public Matrix(int rows, int columns, Double[][] values) {
        this.rows = rows;
        this.columns = columns;
        this.values = new Double[rows][columns];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < columns; j++) {
                this.values[i][j] = values[i][j];
            }
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

    public static Matrix transpose(Matrix first) {
        Double[][] newValues = new Double[first.getColumns()][first.getRows()];
        for (int i = 0; i < first.getRows(); i++) {
            for (int j = 0; j < first.getColumns(); j++) {
                newValues[j][i] = first.getValues()[i][j];
            }
        }
        return new Matrix(first.getColumns(), first.getRows(), newValues);
    }
    public static Matrix add(Matrix first, Matrix second) {
        assert first.getColumns() == second.getColumns() && first.getRows() == second.getRows() :
                String.format("Matrix.add doesn't match for shape (%d, %d) (%d, %d)",
                        first.getRows(), first.getColumns(), second.getRows(), second.getColumns());;

        Double[][] newValues = first.values;
        for (int i = 0; i < first.getRows(); i++) {
            for (int j = 0; j < first.getColumns(); j++) {
                newValues[i][j] += second.values[i][j];
            }
        }
        return new Matrix(first.getRows(), first.getColumns(), newValues);
    }

    public static Matrix dot(Matrix first, Matrix second) {
        assert first.getColumns() == second.getRows() :
                String.format("Matrix.dot doesn't match for shape (%d, %d) (%d, %d)",
                        first.getRows(), first.getColumns(), second.getRows(), second.getColumns());

        int firstRows = first.getRows();
        int secondCols = second.getColumns();
        int firstCols = first.getColumns();
        Double[][] firstValues = first.getValues();
        Double[][] secondValues = second.getValues();
        Double[][] newValues = new Double[firstRows][secondCols];
        for (int i = 0; i < firstRows; i++) {
            for (int j = 0; j < secondCols; j++) {
                double val = 0;
                for (int k = 0; k < firstCols; k++) {
                    val += firstValues[i][k] * secondValues[k][j];
                }
                newValues[i][j] = val;
            }
        }
        return new Matrix(firstRows, secondCols, newValues);
    }

    public static Matrix multiplyElementWise(Matrix first, Matrix second) {
        assert first.getRows() == second.getRows() && first.getColumns() == second.getColumns() :
                String.format("Matrix.multiplyElementWise doesn't match for shape (%d, %d) (%d, %d)",
                        first.getRows(), first.getColumns(), second.getRows(), second.getColumns());

        int firstRows = first.getRows();
        int firstCols = first.getColumns();
        Double[][] newValues = new Double[firstRows][firstCols];
        for (int i = 0; i < firstRows; i++) {
            for (int j = 0; j < firstCols; j++) {
                newValues[i][j] = first.getValues()[i][j] * second.getValues()[i][j];
            }
        }
        return new Matrix(firstRows, firstCols, newValues);
    }

    public static Matrix divideElementWise(Matrix first, Matrix second) {
        assert first.getRows() == second.getRows() && first.getColumns() == second.getColumns() :
                String.format("Matrix.divideElementWise doesn't match for shape (%d, %d) (%d, %d)",
                        first.getRows(), first.getColumns(), second.getRows(), second.getColumns());

        int firstRows = first.getRows();
        int firstCols = first.getColumns();
        Double[][] newValues = new Double[firstRows][firstCols];
        for (int i = 0; i < firstRows; i++) {
            for (int j = 0; j < firstCols; j++) {
                newValues[i][j] = first.getValues()[i][j] / second.getValues()[i][j];
            }
        }
        return new Matrix(firstRows, firstCols, newValues);
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

    public static Matrix identity(int rows, int cols) {
        Matrix out = new Matrix(rows, cols);
        for (int i = 0; i < out.getRows(); i++) {
            for (int j = 0; j < out.getColumns(); j++) {
                if (i == j)
                    out.getValues()[i][j] = 1.0;
                else
                    out.getValues()[i][j] = 0.0;
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

    public static Matrix columnDuplicates(Double[] vector, int numOfCols) {
        Matrix out = new Matrix(vector.length, numOfCols);
        for (int i = 0; i < vector.length; i++) {
            for (int j = 0; j < numOfCols; j++) {
                out.getValues()[i][j] = vector[j];
            }
        }
        return out;
    }

    public static Matrix get1RowVec(int numOfCols) {
        Matrix out = new Matrix(1, numOfCols);
        for (int i = 0; i < numOfCols; i++) {
            out.getValues()[0][i] = 1.0;
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

    public void print() {
        System.out.println(this);
    }

    public Matrix copy() {
        return new Matrix(this.getRows(), this.getColumns(), this.getValues());
    }

    public static void main(String[] args) {
        Matrix matrix = new Matrix(2, 3);

        Matrix matrixT = Matrix.transpose(matrix);
        matrixT.print();
        matrix.print();
        Matrix added = Matrix.dot(matrixT, matrix);
        added.print();
    }
}
