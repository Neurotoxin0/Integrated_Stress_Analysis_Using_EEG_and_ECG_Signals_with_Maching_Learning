/*Author: Xilai Wang*/
/*this service connects the front-end and back-end server. this service is used in userinput component. */
import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';

@Injectable({
  providedIn: 'root'
})
export class RunModelService {

  constructor(private http: HttpClient) { }
  runPythonScript(param: any) {
    //console.log(param);   #param no problem here.
    const url = 'http://127.0.0.1:8000/post/'; 
    return this.http.post(url, {param});
  }
}
