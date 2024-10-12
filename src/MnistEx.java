import Functions.*;
import Matrix.Matrix;
import java.util.ArrayList;
import java.util.List;
import java.io.File;
import java.util.Scanner;

public class MnistEx {

    public static int predict(Model model, Matrix x) {
        Matrix logits = model.feedForward(x);
        int argMax = 0;
        double max = -1;
        for (int j = 0; j < logits.getRows(); j++) {
            if (logits.getValues()[j][0] > max) {
                max = logits.getValues()[j][0];
                argMax = j;
            }
        }
        return argMax;
    }

    public static void main(String[] args) throws Exception {
        Scanner sc = new Scanner(new File("C:\\Users\\nirko\\IdeaProjects\\Neural_network_from_scratch\\src\\datasets\\mnist_train.csv"));
        //parsing a CSV file into the constructor of Scanner class
        sc.useDelimiter("\n");
        List<Matrix> xTrain = new ArrayList<>();
        List<Matrix> yTrain = new ArrayList<>();
        String row;
        String[] separated;
        while (sc.hasNext()) {
            row = sc.next();
            separated = row.split(",");
            Matrix y = new Matrix(10, 1);
            for (int i = 0; i < y.getRows(); i++) {
                y.getValues()[i][0] = i == Integer.parseInt(separated[0]) ? 1.0 : 0.0;
            }
            yTrain.add(y);
            Matrix x = new Matrix(separated.length - 1, 1);
            for (int i = 1; i < separated.length - 1; i++) {
                x.getValues()[i][0] = Double.parseDouble(separated[i]) / 255;
            }
            xTrain.add(x);
        }
        sc.close();

        sc = new Scanner(new File("C:\\Users\\nirko\\IdeaProjects\\Neural_network_from_scratch\\src\\datasets\\mnist_test.csv"));
        //parsing a CSV file into the constructor of Scanner class
        sc.useDelimiter("\n");
        List<Matrix> xTest = new ArrayList<>();
        List<Matrix> yTest = new ArrayList<>();
        while (sc.hasNext()) {
            row = sc.next();
            separated = row.split(",");
            Matrix y = new Matrix(10, 1);
            for (int i = 0; i < y.getRows(); i++) {
                y.getValues()[i][0] = i == Integer.parseInt(separated[0]) ? 1.0 : 0.0;
            }
            yTest.add(y);
            Matrix x = new Matrix(separated.length - 1, 1);
            for (int i = 1; i < separated.length - 1; i++) {
                x.getValues()[i][0] = Double.parseDouble(separated[i]) / 255;
            }
            xTest.add(x);
        }
        sc.close();
        Model model = new Model(16, new CrossEntropy(), 0.01);
        model.add(xTrain.get(0).getRows(), new Relu());
        model.add(128, new Relu());
        model.add(64, new Relu());
//        model.add(64, new Relu());
//        model.add(32, new Relu());
        model.add(32, new Relu());
        model.add(10, new Softmax());

        model.trainModel(xTrain.subList(0, 1000), yTrain.subList(0, 1000), 7, xTest.subList(0, 100), yTest.subList(0, 100), 1);
        model.setLearningRate(0.002);
        model.trainModel(xTrain.subList(0, 1000), yTrain.subList(0, 1000), 1, xTest.subList(0, 100), yTest.subList(0, 100), 1);

        double acc = 0;
        for (int i = 0; i < xTest.size(); i++) {
            int argMax = predict(model, xTest.get(i));
            int trueArg = -1;
            for (int j = 0; j < yTest.get(i).getRows(); j++) {
                if (yTest.get(i).getValues()[j][0] == 1) {
                    trueArg = j;
                }
            }
            System.out.printf("true: %d  predicted: %d%n", trueArg, argMax);
            if (yTest.get(i).getValues()[argMax][0] == 1.0)
                acc += 1.0;
        }
        acc /= xTest.size();
        System.out.println("Accruacy = " + acc);
        model.saveModel("C:\\Users\\nirko\\IdeaProjects\\Neural_network_from_scratch\\src\\mnistModel.ser");
    }
}
