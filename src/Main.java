import org.opencv.core.Mat;

public class Main {

	public static void main(String[] args) {
		// TODO Auto-generated method stub
		//SVM_ svm = new SVM_();
		//svm.myMethod();
//		TrainingDataBuilder dataBuilder = new TrainingDataBuilder("train");
//		double[][] train_data = dataBuilder.getTrainingFeatures();
//		double[] train_labels = dataBuilder.getLabels();
//		SVMHandler.init();
//		SVMHandler.train(train_data, train_labels);
		
		TestDataBuilder testData = new TestDataBuilder("test");
		testData.overAllTest();
//		double[] data = testData.getTestFeatures("1 (1).jpg");
//		double[] result = SVMHandler.test(data);
//		System.out.println("result : "+result[0]);
//		System.out.println("probabilities :: 1:"+result[1]+"   2:"+result[2]);
		System.out.println("success");

	}

}
