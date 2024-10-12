package Functions;

import Matrix.Matrix;

public class Tanh implements Function {
    @Override
    public Matrix applyOn(Matrix input) {
        Matrix out = new Matrix(input.getRows(), input.getColumns());
        for (int i = 0; i < out.getRows(); i++) {
            for (int j = 0; j < out.getColumns(); j++) {
                double val = input.getValues()[i][j];
                out.getValues()[i][j] = (Math.exp(val) - Math.exp(-1 * val)) / (Math.exp(val) + Math.exp(-1 * val));
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
        return Matrix.add(out, Matrix.multiply(Matrix.multiplyElementWise(this.applyOn(input), this.applyOn(input)), -1));
    }

    @Override
    public double gain() {
        return 5.0 / 3;
    }
}
