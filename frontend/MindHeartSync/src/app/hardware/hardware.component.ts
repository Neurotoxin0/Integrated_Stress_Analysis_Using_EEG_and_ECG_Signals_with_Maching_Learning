import { Component } from '@angular/core';
import { RouterModule } from '@angular/router';
import { NzImageModule } from 'ng-zorro-antd/image';

@Component({
  selector: 'app-hardware',
  standalone: true,
  imports: [RouterModule,NzImageModule],
  template:`
  <img id='unavailable' nz-image nzSrc='../../assets/entrance_pics/to_be_developed.webp' nzDisablePreview="true">
  <br>
  <span nz-typography class ='hardwaretext'>Not available yet.</span>
  <br>
  <span nz-typography class ='hardwaretext'>We are working diligently to bring it. Stay tuned for update please.</span>
  <br>
  <span nz-typography class ='hardwaretext'>Thank you for your patience and support.</span>     
  <br>
  <button id='hardwareback' nz-button nzType = 'primary' nzSize='large' nzShape="round" routerLink='/menu' >back</button>
  `,
  styleUrl: './hardware.component.scss'
})
export class HardwareComponent {

}
