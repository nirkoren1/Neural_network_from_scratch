package Functions;

public class None implements Function {
    public double applyOn(double input) {
        return input;
    }
    public double derivativeApplyOn(double input) {
        return 1.0;
    }
}
