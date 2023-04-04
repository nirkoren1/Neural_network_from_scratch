package Functions;

import Matrix.Matrix;

public class Sigmoid implements Function {
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
        return Matrix.add(out, Matrix.multiply(this.applyOn(input), -1));
    }

    @Override
    public double gain() {
        return 1.0;
    }
}
