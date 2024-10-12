package Functions;

import Matrix.Matrix;

import java.io.Serializable;

public class Relu implements Function, Serializable {
    public Matrix applyOn(Matrix input) {
        Matrix out = new Matrix(input.getRows(), input.getColumns());
        for (int i = 0; i < out.getRows(); i++) {
            for (int j = 0; j < out.getColumns(); j++) {
                out.getValues()[i][j] = Math.max(0.0, input.getValues()[i][j]);
            }
        }
        return out;
    }
    public Matrix derivativeApplyOn(Matrix input) {
        Matrix out = new Matrix(input.getRows(), input.getColumns());
        for (int i = 0; i < out.getRows(); i++) {
            for (int j = 0; j < out.getColumns(); j++) {
                if (input.getValues()[i][j] <= 0)
                    out.getValues()[i][j] = 0.0;
                else
                    out.getValues()[i][j] = 1.0;
            }
        }
        return out;
    }

    @Override
    public double gain() {
        return Math.sqrt(2);
    }

    public static void main(String[] args) {
        Matrix ex = new Matrix(3, 2);
        ex.print();
        Relu relu = new Relu();
        relu.applyOn(ex).print();
    }
}
