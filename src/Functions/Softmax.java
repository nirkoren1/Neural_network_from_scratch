package Functions;

import Matrix.Matrix;

import java.io.Serializable;

public class Softmax implements Function, Serializable {

    @Override
    public Matrix applyOn(Matrix input) {
        double sum = 0;
        double max = -1e10;
        for (int i = 0; i < input.getRows(); i++) {
            for (int j = 0; j < input.getColumns(); j++) {
                if (input.getValues()[i][j] > max)
                    max = input.getValues()[i][j];
            }
        }
        for (int i = 0; i < input.getRows(); i++) {
            for (int j = 0; j < input.getColumns(); j++) {
                sum += Math.exp(input.getValues()[i][j] - max);
            }
        }
        Matrix out = new Matrix(input.getRows(), input.getColumns());
        for (int i = 0; i < out.getRows(); i++) {
            for (int j = 0; j < out.getColumns(); j++) {
               out.getValues()[i][j] = Math.exp(input.getValues()[i][j] - max) / sum;
            }
        }
        return out;
    }

    @Override
    public Matrix derivativeApplyOn(Matrix input) {
        Matrix out;
        out = Matrix.dot(input, Matrix.trancePose(input));
        out = Matrix.multiply(out, -1);
        Matrix I = Matrix.identity(input.getRows(), input.getRows());
        out = Matrix.add(I, out);
        return out;
    }

    @Override
    public double gain() {
        return 1.0;
    }

    public static void main(String[] args) {

    }
}
