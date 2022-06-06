package Functions;

import Matrix.Matrix;

public interface CostFunction {
    public Matrix cost(Matrix predicted, Matrix realValue);
}
