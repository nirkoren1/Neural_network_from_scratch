package Functions;

public class Sigmoid implements Function {
    @Override
    public double applyOn(double input) {
        return 1 / (1 + Math.exp(-1 * input));
    }

    @Override
    public double derivativeApplyOn(double input) {
        return 1 - this.applyOn(input);
    }
}
