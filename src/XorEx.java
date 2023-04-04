import Functions.MSE;
import Functions.Relu;
import Functions.Sigmoid;
import Functions.Tanh;
import Matrix.Matrix;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class XorEx {
    private static Random random = new Random(5);
    public static void main(String[] args) {
        List<Matrix> X = new ArrayList<>();
        List<Matrix> Y = new ArrayList<>();
        for (int i = 0; i < 1000; i++) {
            Matrix x = new Matrix(2, 1);
            for (int j = 0; j < 2; j++) {
                x.getValues()[j][0] = (double) random.nextInt(2);
            }
            X.add(x);
            Matrix y = new Matrix(1, 1);
            if (Math.abs(x.getValues()[0][0] - x.getValues()[1][0]) == 0.0) {
                y.getValues()[0][0] = 0.0;
            } else {
                y.getValues()[0][0] = 1.0;
            }
            Y.add(y);
        }

        List<Matrix> XTest = new ArrayList<>();
        List<Matrix> YTest = new ArrayList<>();
        int testSize = 20;
        for (int i = 0; i < testSize; i++) {
            int rand = random.nextInt(1000);
            XTest.add(X.get(rand));
            YTest.add(Y.get(rand));
        }

        Model model = new Model(16, new MSE(), 0.05);
        model.add(2, new Relu());
        model.add(2, new Tanh());
//        model.add(16, new Relu());
//        model.add(16, new Sigmoid());
        model.add(1, new Relu() );

        model.trainModel(X, Y, 10000, XTest, YTest, 1);
        System.out.println("----------TEST-----------");
        double totalError = 0;
        double accuracy = 0;
        for (int i = 0; i < testSize; i++) {
            System.out.println(XTest.get(i));
            System.out.println(YTest.get(i));
            System.out.println(model.feedForward(XTest.get(i)));
            System.out.println("---------------------");
            totalError += Math.pow(model.feedForward(XTest.get(i)).getValues()[0][0] - YTest.get(i).getValues()[0][0], 2);
            if (Math.abs(model.feedForward(XTest.get(i)).getValues()[0][0] - YTest.get(i).getValues()[0][0]) < 0.5) {
                accuracy += 1;
            }
        }
        totalError /= testSize;
        accuracy /= testSize;
        System.out.println("Total error = " + totalError);
        System.out.println("Accuracy = " + accuracy);
    }
}