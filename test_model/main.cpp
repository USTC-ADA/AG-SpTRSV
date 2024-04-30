#include <Python.h>
#include <string>
#include <sstream>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

using namespace std;

#define duration(a, b) (1.0 * (b.tv_usec - a.tv_usec + (b.tv_sec - a.tv_sec) * 1.0e6))

#define VALUE_TYPE double

void AG_read(char *input_name);
void AG_preprocessing(int ps, int ss, int rb);
void AG_preprocessing2(int *scheme);
void AG_matrix_reorder();
void AG_finalize();
void AG_execute(const char *matrix_name, const char *csv_name);
void AG_matrix_output(char* matrix_info_name, char* matrix_name);
void AG_matrix_info(float *args, int size);
void AG_new_ana();

#define PY PyRun_SimpleString

int main(int argc, char* argv[])
{

    int ch;

    int input_flag = 0, outcsv_flag = 0, model_flag = 0, prep_flag = 0;
    char *input_name, *outcsv_name, *model_name, *prep_name;

    while ((ch = getopt(argc, argv, "i:o:m:p:")) != -1)
    {
        switch (ch)
        {
            case 'i':
                input_flag = 1;
                input_name = optarg;
                break;

            case 'o':
                outcsv_flag = 1;
                outcsv_name = optarg;
                break;

            case 'm':
                model_flag = 1;
                model_name = optarg;
                break;
            
            case 'p':
                prep_flag = 1;
                prep_name = optarg;
            
        }
    }

    if (input_flag == 0 || model_flag == 0)
    {
        printf("[Usage]: ./main -i {input_filename} -m {model_filename} -o {output_filename}\n");
        exit(1);
    }

    struct timeval tv_begin, tv_end, tv_begin2, tv_end2;
    float model_run_time = 0.0;
    float prep_time = 0.0;

    Py_Initialize();

    stringstream ss;
    ss << "sys.path.append(\"" << model_name << "\")";
    string model_append = ss.str();

    PY("import sys");
    PY("sys.path.append(\"../model\")");
    // PY(model_append.c_str());

    PyObject* pModule = PyImport_ImportModule("model_predict");
	if( pModule == NULL )
    {
		printf("Module 'model_predict' not found\n");
		return 1;
	}

    PyObject* pModule2 = PyImport_ImportModule("mlp_model_def");
	if( pModule2 == NULL )
    {
		printf("Module 'mlp_model_def' not found\n");
		return 1;
	}

    PyObject* Mload = PyObject_GetAttrString(pModule, "module_load");
	if( Mload == NULL || !PyCallable_Check(Mload))
    {
		printf("Callable function not found\n");
		return 1;
	}

    PyObject* pyParams_model_name = Py_BuildValue("s", model_name);
    PyObject* args_model_name = PyTuple_New(1);
    PyTuple_SetItem(args_model_name, 0, pyParams_model_name);
    PyObject* ret = PyObject_CallObject(Mload, args_model_name);
    if (ret == NULL)
    {
        printf("Failed function call\n");
        return 1;
    }

    int m, nnz;
    float avg_rnnz, avg_lnnz;
    float cov_rnnz, cov_lnnz;
    float dep_dist, reverse;

    printf("Matrix name: %s\n", input_name);

    AG_read(input_name);
    AG_new_ana();

    gettimeofday(&tv_begin, NULL);

    const int size = 11;
    float out_args[size];
    AG_matrix_info(out_args, size);

    PyObject* Mpred = PyObject_GetAttrString(pModule, "model_predict");
	if( Mpred == NULL || !PyCallable_Check(Mpred))
    {
		printf("Callable function not found\n");
		return 1;
	}

    gettimeofday(&tv_begin2, NULL);

    PyObject* pyParams = PyList_New(0);
    for (int i = 0; i < 10; i++)
        PyList_Append(pyParams, Py_BuildValue("f", out_args[i]));
    PyObject* args = PyTuple_New(1);
    PyTuple_SetItem(args, 0, pyParams);
    PyEval_CallObject(Mpred, args);
    PyObject* pyValue = PyObject_CallObject(Mpred, args);

    gettimeofday(&tv_end2, NULL);
    printf("Model run time %.2f us\n", duration(tv_begin2, tv_end2));
    model_run_time = duration(tv_begin2, tv_end2);

    int scheme[8];
    for (int i = 0; i < 8; i++)
    {
        PyObject *Item = PyList_GetItem(pyValue, i);
        PyArg_Parse(Item, "i", &scheme[i]);
        Py_DECREF(Item);
    }

    gettimeofday(&tv_end, NULL);
    prep_time += duration(tv_begin, tv_end);

    Py_Finalize();

    printf("Selected scheme:\n");
    for (int i = 0; i < 8; i++)
        printf("%d ", scheme[i]);
    printf("\n");
    
    gettimeofday(&tv_begin, NULL);

    AG_preprocessing2(scheme);

    gettimeofday(&tv_end, NULL);

    prep_time += duration(tv_begin, tv_end);

    if (outcsv_flag)
        AG_execute(input_name, outcsv_name);
    else
        AG_execute(input_name, NULL);

    if (prep_flag)
    {
        FILE *fp = fopen(prep_name, "a");
        fprintf(fp, "%s,%.2f,%.2f\n", input_name, model_run_time, prep_time);
    }

}