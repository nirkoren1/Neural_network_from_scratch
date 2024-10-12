package Functions;

import Matrix.Matrix;

import java.io.Serializable;

public class Sigmoid implements Function, Serializable {
    @Override
    public Matrix applyOn(Matrix input) {
        Matrix out = new Matrix(input.getRows(), input.getColumns());
        for (int i = 0; i < out.getRows(); i++) {
            for (int j = 0; j < out.getColumns(); j++) {
                out.getValues()[i][j] = 1 / (1 + Math.exp(-1 * input.getValues()[i][j]));
            }
        }
        return out;
    }

    @Override
    public Matrix derivativeApplyOn(Matrix input) {
        Matrix out = new Matrix(input.getRows(), input.getColumns());
        for (int i = 0; i < out.getRows(); i++) {
            for (int j = 0; j < out.getColumns(); j++) {
                out.getValues()[i][j] = 1.0;
            }
        }
        Matrix applied = this.applyOn(input);
        return Matrix.multiplyElementWise(applied, Matrix.add(out, Matrix.multiply(applied, -1)));
    }

    @Override
    public double gain() {
        return 1.0;
    }
}
