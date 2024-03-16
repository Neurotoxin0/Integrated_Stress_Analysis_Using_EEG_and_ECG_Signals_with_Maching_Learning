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
export class AfterSubmissionDialogComponent {
  constructor(@Inject(MAT_DIALOG_DATA) public data: any) { }
}
