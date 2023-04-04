package Functions;

import Matrix.Matrix;

public interface Function {
    public Matrix applyOn(Matrix input);
    public Matrix derivativeApplyOn(Matrix input);
    public double gain();
}
