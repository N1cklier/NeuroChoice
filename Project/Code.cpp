// Вас приветствует нейросеть "Нейровыбор"!
// Строки с числовым комментарием ( "//" ) - строки, которые необходимо включить или выключить для работы других вариаций программы.
// Варианты программы: 1) процентный показ резульатов; 2) один результат; 3) средний процент верных ответов на тестовую базу; 4) процент правильных ответов на одну тестовую бащу.
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <cmath>
#include <ctime>
#include <map> // 1
using namespace std;
const short int k1 = 10, ex = 20, k2 = 12; // k1 - кол-во профессий, ex - кол-во примеров, k2 - кол-во вопросов
unsigned short int traindata[k1 * ex][k2], resultdata[k1 * ex];
short int arr[k2], v; // 1, 2
void read() {
	string file = "Data.txt";
	ifstream reading(file);
	if (reading) {
		for (int i = 0; i < k1 * ex; i++) {
			reading >> resultdata[i];
			for (int k = 0; k < k2; k++)
				reading >> traindata[i][k];
		}
		for (int i = 0; i < k1 * ex; i++) {
			int r = rand() % (k1 * ex);
			for (int j = 0; j < k2; j++)
				swap(traindata[i][j], traindata[r][j]);
			swap(resultdata[i], resultdata[r]);
		}
	}
	else cout << "Не удалось открыть файл " << file << endl;
	reading.close();
}
double sigmoid(double rec) {
	return 1. / (1 + exp(-rec));
}
double der_sigmoid(double rec) {
	return sigmoid(rec) * (1 - sigmoid(rec));
}
double cross_entropy(double* proc, int res) {
	return -log(proc[res]);
}
void test() { // 1, 2
start:
	cout << "Выберите тип ввода: 1) тест (рекомендуется); 2) сплошной (без лишних слов)." << endl;
	cin >> v;
	if (v == 1) {
		int k = 0;
		cout << "На каждом следующем критерие, характиризующем вас, укажите число от 0 до 2, где\n0 - не относится,\n1 - затрудняюсь ответить,\n2 - относится\n";
		cout << "Интроверсия - "; cin >> arr[k]; k++;
		cout << "На каждом следующем критерие укажите число от 0 до 3, где\n0 - абсолютно не заинтересован,\n1 - слабо заинтересован,\n2 - достаточно заинтересован,\n3 - представляет наибольший интерес\n";
		cout << "Биология: "; cin >> arr[k]; k++;
		cout << "География: "; cin >> arr[k]; k++;
		cout << "Иностранные языки: "; cin >> arr[k]; k++;
		cout << "Информатика: "; cin >> arr[k]; k++;
		cout << "История: "; cin >> arr[k]; k++;
		cout << "Литература: "; cin >> arr[k]; k++;
		cout << "Математика: "; cin >> arr[k]; k++;
		cout << "Обществознание: "; cin >> arr[k]; k++;
		cout << "Физика: "; cin >> arr[k]; k++;
		cout << "Физическая подготовка: "; cin >> arr[k]; k++;
		cout << "Химия: "; cin >> arr[k]; k++;
	}
	else if (v == 2)
		for (int i = 0; i < k2; i++)
			cin >> arr[i];
	else {
		cout << "Такой вариант не представлен";
		goto start;
	}
	if (arr[0] < 0 || arr[0] > 2) {
		cout << "Не все входные данные соответствуют запросу. Проверьте их и попробуйте ещё раз.\n";
		goto start;
	}
	for (unsigned short int i = 1; i < k2; i++) {
		if (arr[i] < 0 || arr[i] > 3) {
			cout << "Не все входные данные соответствуют запросу. Проверьте их и попробуйте ещё раз.\n";
			goto start;
		}
	}
}
int main() {
	setlocale(LC_ALL, "RUS");
	srand(time(0));
	//int persum = 0; // 3
	//int perkol = 0; // 3
	read();
	test(); // 1, 2
	int res[k1] = {}; // 1
	for (int l = 0; l < 50; l++) { // 1, 3
		const unsigned short int datasize = k1 * ex;
		const unsigned short int insize = k2 + 1;
		const unsigned short int outsize = k1 + 1;
		unsigned short int hidesize = 2 * insize / 3 + 7;
		double** w1 = new double* [insize];
		for (int i = 0; i < insize; i++)
			w1[i] = new double[hidesize];
		double** w2 = new double* [hidesize];
		for (int i = 0; i < hidesize; i++)
			w2[i] = new double[outsize];
		double* bias1 = new double[insize];
		double* bias2 = new double[outsize];
		for (int i = 0; i < hidesize; i++) {
			bias1[i] = (double)rand() / RAND_MAX;
			for (int j = 0; j < insize; j++)
				w1[j][i] = (double)rand() / RAND_MAX;
		}
		for (int i = 0; i < outsize; i++) {
			bias2[i] = (double)rand() / RAND_MAX;
			for (int j = 0; j < hidesize; j++)
				w2[j][i] = (double)rand() / RAND_MAX;
		}
		double* rec1 = new double[hidesize]; // полученный сигнал (receive)
		double* rec2 = new double[outsize];
		double* proc1 = new double[hidesize]; // обработанный сигнал (process)
		double* proc2 = new double[outsize];
		double* der_rec1 = new double[hidesize]; // производная (derivative)
		double* der_rec2 = new double[outsize];
		double* der_proc1 = new double[hidesize];
		double** der_w1 = new double* [insize];
		for (int i = 0; i < insize; i++)
			der_w1[i] = new double[hidesize];
		double** der_w2 = new double* [hidesize];
		for (int i = 0; i < hidesize; i++)
			der_w2[i] = new double[outsize];
		double* der_b1 = new double[hidesize];
		double* der_b2 = new double[outsize];
		float rate = 0.011; // скорость
		unsigned short int epochs = 17; // количество эпох
		double error;
		for (unsigned short int ep = 0; ep < epochs; ep++) {
			for (unsigned short int ds = 0; ds < datasize; ds++) {
				for (unsigned short int i = 0; i < hidesize; i++) {
					rec1[i] = 0;
					for (unsigned short int j = 0; j < insize; j++)
						rec1[i] += traindata[ds][j] * w1[j][i];
					rec1[i] += bias1[i];
					proc1[i] = sigmoid(rec1[i]);
				}
				double sum = 0;
				for (unsigned short int i = 0; i < outsize; i++) {
					rec2[i] = 0;
					for (unsigned short int j = 0; j < hidesize; j++)
						rec2[i] += proc1[j] * w2[j][i];
					rec2[i] += bias2[i];
					sum += exp(rec2[i]);
				}
				for (unsigned short int i = 0; i < outsize; i++)
					proc2[i] = exp(rec2[i]) / sum;
				error = cross_entropy(proc2, resultdata[ds]);
				for (unsigned short int i = 0; i < outsize; i++) {
					der_rec2[i] = proc2[i];
					if (i == resultdata[ds]) der_rec2[i]--;
					for (unsigned short int j = 0; j < hidesize; j++)
						der_w2[j][i] = proc1[j] * der_rec2[i];
					der_b2[i] = der_rec2[i];
				}
				for (unsigned short int i = 0; i < hidesize; i++) {
					der_proc1[i] = 0;
					for (unsigned short int j = 0; j < outsize; j++)
						der_proc1[i] += der_rec2[j] * w2[i][j];
					der_rec1[i] = der_proc1[i] * der_sigmoid(rec1[i]);
				}
				for (unsigned short int i = 0; i < hidesize; i++) {
					for (unsigned short int j = 0; j < insize; j++)
						der_w1[j][i] = traindata[ds][j] * der_rec1[i];
					der_b1[i] = der_rec1[i];
				}
				for (unsigned short int i = 0; i < hidesize; i++) {
					for (unsigned short int j = 0; j < insize; j++)
						w1[j][i] -= rate * error * der_w1[j][i];
					der_b1[i] -= rate * error * der_b1[i];
				}
				for (unsigned short int i = 0; i < outsize; i++) {
					for (unsigned short int j = 0; j < hidesize; j++)
						w2[j][i] -= rate * error * der_w2[j][i];
					der_b2[i] -= rate * error * der_b2[i];
				}
			}
		}
		//начало 3, 4
		//int correct = 0;
		//for (unsigned short int ds = 0; ds < datasize; ds++) {
		//	for (unsigned short int i = 0; i < hidesize; i++) {
		//		rec1[i] = 0;
		//		for (unsigned short int j = 0; j < insize; j++) 
		//			rec1[i] += traindata[ds][j] * w1[j][i];
		//		rec1[i] += bias1[i];
		//		proc1[i] = sigmoid (rec1[i]);
		//	}
		//	double sum = 0;
		//	for (unsigned short int i = 0; i < outsize; i++) {
		//		rec2[i] = 0;
		//		for (unsigned short int j = 0; j < hidesize; j++) 
		//			rec2[i] += proc1[j] * w2[j][i];
		//		rec2[i] += bias2[i];
		//		sum += exp (rec2[i]);
		//	}
		//	int imax = -1;
		//	double procmax = -1;
		//	for (unsigned short int i = 0; i < outsize; i++) {
		//		proc2[i] = exp (rec2[i]) / sum;
		//		if (procmax < proc2[i]) {
		//			imax = i;
		//			procmax = proc2[i];
		//		}
		//	}
		//	if (resultdata[ds] == imax) correct++;
		//}
		//конец 3, 4
		//perkol++; // 3
		//persum += int (float(correct) / float(datasize) * 100); // 3
		////cout << "Верно " << correct << " из " << datasize << ", " << int(float(correct) / float(datasize) * 100) << "%" << endl; // 4
		//} // 3
		//cout << "Средний процент правильных ответов составляет: " << persum / perkol << endl; // 3
		for (unsigned short int i = 0; i < hidesize; i++) {
			rec1[i] = 0;
			for (unsigned short int j = 0; j < insize; j++)
				rec1[i] += arr[j] * w1[j][i];
			rec1[i] += bias1[i];
			proc1[i] = sigmoid(rec1[i]);
		}
		double sum = 0;
		for (unsigned short int i = 0; i < outsize; i++) {
			rec2[i] = 0;
			for (unsigned short int j = 0; j < hidesize; j++)
				rec2[i] += proc1[j] * w2[j][i];
			rec2[i] += bias2[i];
			sum += exp(rec2[i]);
		}
		int imax = -1;
		double procmax = -1;
		for (unsigned short int i = 0; i < outsize; i++) {
			proc2[i] = exp(rec2[i]) / sum;
			if (procmax < proc2[i]) {
				imax = i;
				procmax = proc2[i];
			}
		}
		res[imax - 1]++; // 1
		//начало 2
		//string result[k1] = {
		//"Артист", "Дизайнер", "Журналист", "Инженер",
		//"Врач", "Переводчик", "Полицейский",
		//"Преподаватель", "Строитель", "IT-специалист" 
		//};
		//test();
		//cout << "Вам может подойти следующее направление: " << result[imax - 1] << endl;
		//конец 2
	} // 1
	//начало 1
	string result[k1] = {
		"Артист", "Дизайнер", "Журналист", "Инженер",
		"Врач", "Переводчик", "Полицейский", 
		"Преподаватель", "Строитель", "IT-специалист" 
		}; 
	map<int, int> mp; 
	for (int i = 0; i < k1; i++)
		if (res[i] != 0) mp.emplace(res[i], i); 
	cout << "Вам могут подойти следующие специальности:\n"; 
	for (auto i = mp.rbegin(); i != mp.rend(); i++) 
		cout << result[i->second] << " - " << 1. * res[i->second] / 50 * 100 << "%\n"; 
	//конец 1
	return 0;
}