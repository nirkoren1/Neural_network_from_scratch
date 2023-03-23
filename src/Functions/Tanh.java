package Functions;

public class Tanh implements Function {
    @Override
    public double applyOn(double input) {
        return (Math.exp(input) - Math.exp(-1 * input)) / (Math.exp(input) + Math.exp(-1 * input));
    }

    @Override
    public double derivativeApplyOn(double input) {
        return 1 - Math.pow(this.applyOn(input), 2);
    }
}
