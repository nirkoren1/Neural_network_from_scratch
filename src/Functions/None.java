package Functions;

import Matrix.Matrix;

public class None implements Function {
    public Matrix applyOn(Matrix input) {
        return input.copy();
    }
    public Matrix derivativeApplyOn(Matrix input) {
        return Matrix.transpose(Matrix.get1RowVec(input.getRows()));
    }

    @Override
    public double gain() {
        return 1;
    }
}
