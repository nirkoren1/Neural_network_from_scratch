package Functions;

import Matrix.Matrix;

public class MSE implements CostFunction {
    public Matrix cost(Matrix predicted, Matrix realValue) {
        return Matrix.multiply(Matrix.trancePose(Matrix.add(predicted, Matrix.multiply(realValue, -1))),
                2 / predicted.getRows());
    }
}
