package Functions;

public class Relu implements Function {
    public double applyOn(double input) {
        if (input <= 0)
            return 0;
        return input;
    }
    public double derivativeApplyOn(double input) {
        if (input < 0)
            return 0;
        return 1;
    }
}
