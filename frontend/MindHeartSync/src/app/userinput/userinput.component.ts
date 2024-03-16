/*Author: Xilai Wang*/
/*This is the component related to the input page. this component accepts user input and send it to the Django back-end and let back-end
do prediction. The back-end response the result to this component and this component will process the result and display it to user.*/
import { Component, Input, OnInit } from '@angular/core';
import { ActivatedRoute } from '@angular/router';
import { NzIconModule } from 'ng-zorro-antd/icon';
import { NzLayoutModule } from 'ng-zorro-antd/layout';
import { NzMenuModule } from 'ng-zorro-antd/menu';
import { NzImageModule } from 'ng-zorro-antd/image';
import { NzAvatarModule } from 'ng-zorro-antd/avatar';
import { NzCarouselModule } from 'ng-zorro-antd/carousel';
import { NzButtonModule } from 'ng-zorro-antd/button';
import { NzTypographyModule } from 'ng-zorro-antd/typography';
import { RouterModule, RouterOutlet } from '@angular/router';
import { NzCardModule } from 'ng-zorro-antd/card';
import { NzGridModule } from 'ng-zorro-antd/grid';
import { NzDividerModule } from 'ng-zorro-antd/divider';
import { NzInputModule } from 'ng-zorro-antd/input';
import { Form, FormsModule } from '@angular/forms';
import { NzDropDownModule } from 'ng-zorro-antd/dropdown';
import { NzSelectModule } from 'ng-zorro-antd/select';
import { NzRadioModule } from 'ng-zorro-antd/radio';
import { NzPageHeaderModule } from 'ng-zorro-antd/page-header';
import { RunModelService } from '../run-model.service';
import { HttpClientModule } from '@angular/common/http';
import { Router } from '@angular/router';
import { MatDialog } from '@angular/material/dialog';
import { AfterSubmissionDialogComponent } from '../after-submission-dialog/after-submission-dialog.component';

@Component({
  selector: 'app-userinput',
  standalone: true,
  imports: [NzDividerModule, NzInputModule, FormsModule, NzButtonModule, NzRadioModule,NzPageHeaderModule,HttpClientModule,AfterSubmissionDialogComponent],
  providers:[RunModelService,MatDialog,],
  templateUrl: './userinput.component.html',
  styleUrl: './userinput.component.scss'
})
export class UserinputComponent implements OnInit {
  myResponseData: any;
  inputValue?: string;
  radioValue :string = 'mlp';
  
  //construction function of the component, initializing router, httpservice and dialog objects.
  constructor(private route: ActivatedRoute, private runModelService: RunModelService,private router: Router,public dialog: MatDialog) {}
  //get which model did user choose in the previous page from router snapshot.
  ngOnInit() {  
    this.radioValue = this.route.snapshot.paramMap.get('chosen_model')||'mlp';
  }
  //Define the feedback dialog.
  openDialog(content:string,icon:string): void {
    const dialogRef = this.dialog.open(AfterSubmissionDialogComponent, {
      width: '800px',
      data: { title: 'Your Stress Level', content: content, icon:icon }
    });

    dialogRef.afterClosed().subscribe(result => {
      //console.log('The dialog was closed');
    });
  }
  //this function is everything that front-end do after user clicks the submit button.
  submit_data(model:string, ecg:string, eeg_delta:string, eeg_theta:string, eeg_alpha:string, eeg_beta1:string, eeg_beta2:string, eeg_gamma1:string, eeg_gamma2:string){
    //1. judge if the input is valid using Regex.
    // if valid, keep going on next step, if invalid, pop up a alarm to let user edit.

    let eeg = [eeg_delta, eeg_theta, eeg_alpha, eeg_beta1, eeg_beta2, eeg_gamma1, eeg_gamma2];
    let eeg_data_freq = ['Delta', 'Theta', 'Alpha', 'Beta 1', 'Beta 2', 'Gamma 1','Gamma 2'];
    const regexECG = /^\s*([+-]?\d*(\.\d*)?\s+){10}([+-]?\d*(\.\d*)?\s*)$/;
    const regexEEG = /^\s*([+-]?\d*(\.\d*)?\s+){7}([+-]?\d*(\.\d*)?\s*)$/;
    //test the format of ECG data
    if(regexECG.test(ecg) == false){
      alert('Incorrect format of input of ECG data!');
      return;
    }
    //test the format of EEG data
    for(let i = 0; i<=6; i++){
      if(regexEEG.test(eeg[i]) == false){
        alert('Incorrect format of input of EEG data in frequency 【'+ eeg_data_freq[i]+'】!');
        return;
      }
    }
    //2. send the data to back-end
    //extract a float array from the strings.
    const floatsRegex = /[-+]?[0-9]*\.?[0-9]+/g;
    var input_features:number[] = [];
    //handle ECG data and append it to the input_features.
    var matches = ecg.match(floatsRegex);
    if(matches != null){ //in fact matches cannot be null because we have already checked the format of input in step 1.
      let floatsArray = matches.map((match: string) => parseFloat(match.trim()));
      input_features = input_features.concat(floatsArray);
    }
    //handle EEG data and append it to the input_features.
    for(let i = 0;i <= 6; i++){
      matches = eeg[i].match(floatsRegex);
      if(matches!=null){
        let floatsArray = matches.map((match: string) => parseFloat(match.trim()));
        input_features = input_features.concat(floatsArray);
      }
    }
    //this is the data in JSON format that will be sent to the Django server.
    let json_input = JSON.stringify({
      "model_name":model,
      "input_features":input_features
    })
    //dealing with response from Django back-end.
    this.runModelService.runPythonScript(json_input).subscribe({
      next: (response) => {
        this.myResponseData = response;
        //3. pop up a alarm telling user to check history for result and navigate back to menu.
        var stress_level = "";
        let icon = "";
        // after getting the result, decide what to tell user and the icon shown.
        switch(this.myResponseData.result.trim()){
          case "['EO']":
            stress_level = "relaxed";
            icon = "../../assets/emotion_icons/face-grin-wide-solid.svg";
            break;
          case "['AC1']":
            stress_level = "a little bit stressed";
            icon = "../../assets/emotion_icons/face-meh-solid.svg";
            break;
          case "['AC2']":
            stress_level = "stressed";
            icon = "../../assets/emotion_icons/face-tired-solid.svg";
            break;
        }

        let wordsInDialog = `According to the model's prediction, you are ${stress_level} right now.\nYou can check your previous assessments in the history.\nNow we will head you back to the menu.`;
        this.openDialog(wordsInDialog,icon);
        //after showing user the result, navigate back to menu.
        this.router.navigate(['/menu']);
      },
      //if an error occur, output the error to console.
      error: (error) => console.error(error),
    });

    
  }
  
}
