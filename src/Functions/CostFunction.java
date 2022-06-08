package Functions;

import Matrix.Matrix;

public interface CostFunction {
    public double cost(Matrix predicted, Matrix realValue);
    public Matrix derCost(Matrix predicted, Matrix realValue);
}
