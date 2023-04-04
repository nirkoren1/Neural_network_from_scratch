package Functions;

import Matrix.Matrix;

import java.io.Serializable;

public class CrossEntropy implements CostFunction, Serializable {

    @Override
    public double cost(Matrix predicted, Matrix realValue) {
        double sum = 0;
        for (int i = 0; i < predicted.getRows(); i++) {
            sum += realValue.getValues()[i][0] * Math.log(predicted.getValues()[i][0]);
        }
        return -1 * sum;
    }

    @Override
    public Matrix derCost(Matrix predicted, Matrix realValue) {
        return Matrix.multiply(Matrix.divideElementWise(realValue, predicted), -1);
    }
}
