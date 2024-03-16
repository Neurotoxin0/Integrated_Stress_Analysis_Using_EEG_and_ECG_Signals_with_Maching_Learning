'''Author: Xilai Wang'''
from django.shortcuts import render
from django.http import JsonResponse,HttpResponse
import subprocess
from django.views.decorators.csrf import csrf_exempt
import json
import os
# this view is the link between Angular front-end and the machine learning models.
@csrf_exempt    # don't do this in industry!
def deal_post_request(request):
    '''#code for getting relative path of the script who runs model. will be reused if file structure of the project changes
    print('===============cwd===============')
    print(os.getcwd())
    abs_path = 'E:/Integrated_Stress_Analysis_Using_EEG_and_ECG_Signals_with_Maching_Learning/MLalgorithm/run_model.py'
    rela_path = os.path.relpath(abs_path,os.getcwd())
    print(rela_path)
    '''
    relative_path = '../MLalgorithm/run_model.py'
    #get user input from front-end.
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            data = data['param']
            data = json.loads(data)
            model_name = data['model_name']
            input_features = data['input_features']
            #parameter sent to the subprocess is in JSON format
            param = {
                "model_name": model_name,
                "input_features": input_features
            }
            json_param = json.dumps(param)
            #run the .py script which runs the model and capture its output (printout). The output is returned to the varable result.
            result = subprocess.run(['python', relative_path, json_param], 
                                    capture_output=True, text=True)
            '''for testing uses.
            print('-----------------================')
            print(result.stdout)
            '''
            return JsonResponse({'success': True, 'result':result.stdout})
        except json.JSONDecodeError:
            return HttpResponse('Invalid JSON', status=400)
    else:
        return HttpResponse('Only POST method is accepted', status=405)


