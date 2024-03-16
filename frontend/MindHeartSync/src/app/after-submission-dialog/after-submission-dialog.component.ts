/*Author: Xilai Wang*/
/*This component is the dialogue which pop up and shows the predicted stress level to user after user submitted the ECG and EEG data.*/
import { Component, Inject } from '@angular/core';
import { MatDialogModule } from '@angular/material/dialog';
import { MatButtonModule } from '@angular/material/button';
import { MatIconModule } from '@angular/material/icon';
import { MAT_DIALOG_DATA } from '@angular/material/dialog';

@Component({
  selector: 'app-after-submission-dialog',
  standalone: true,
  imports: [MatDialogModule,MatButtonModule, MatIconModule,],
  templateUrl: './after-submission-dialog.component.html',
  styleUrl: './after-submission-dialog.component.scss'
})
/*This dialog component is called by the userinput component when front-end get the response from Django back-end. This dialog 
intakes a string and a path of the emotional icon which represents a stress level. */
export class AfterSubmissionDialogComponent {
  constructor(@Inject(MAT_DIALOG_DATA) public data: any) { }
}
