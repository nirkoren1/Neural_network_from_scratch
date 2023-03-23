package Functions;

import Matrix.Matrix;

public class MSE implements CostFunction {
    public Matrix derCost(Matrix predicted, Matrix realValue) {
        return Matrix.multiply(Matrix.add(predicted, Matrix.multiply(realValue, -1)), 2 / predicted.getRows());
    }
    public double cost(Matrix predicted, Matrix realValue) {
        Matrix out = Matrix.add(predicted, Matrix.multiply(realValue, -1));
        double error = 0;
        for (int i = 0; i < out.getRows(); i++) {
            error += Math.pow(out.getValues()[i][0], 2);
        }
        error /= predicted.getRows();
        return error;
    }
}
