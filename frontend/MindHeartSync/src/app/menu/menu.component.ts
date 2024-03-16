import { Component } from '@angular/core';
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

@Component({
  selector: 'app-menu',
  standalone: true,
  imports: [NzLayoutModule,NzCardModule,NzGridModule,RouterModule,RouterOutlet],
  template: `
  <nz-layout>
    <nz-content>
    <div nz-row nzGutter="128">
      <div nz-col nzSpan="8"><nz-card nzHoverable  [nzCover]="coverTemplate1" routerLink="/model-selection">
      <nz-card-meta nzTitle="Manual Input" nzDescription="Enter EEG and ECG data manually."></nz-card-meta>
    </nz-card>
    <ng-template #coverTemplate1>
      <img src="../../assets/entrance_pics/manual_input.webp" />
    </ng-template></div>
    <div nz-col nzSpan="8"><nz-card nzHoverable  [nzCover]="coverTemplate2" routerLink='/hardware'>
      <nz-card-meta nzTitle="Hardware Connection" nzDescription="Connect device for automatic data input."></nz-card-meta>
    </nz-card>
    <ng-template #coverTemplate2>
      <img src="../../assets/entrance_pics/hardware_connection.webp" />
    </ng-template></div>
    <div nz-col nzSpan="8"><nz-card nzHoverable  [nzCover]="coverTemplate3" routerLink='/hardware'>
      <nz-card-meta nzTitle="History" nzDescription="Browse your assessment history."></nz-card-meta>
    </nz-card>
    <ng-template #coverTemplate3>
      <img src="../../assets/entrance_pics/history.webp" />
    </ng-template></div>
    </div>
    </nz-content>
  </nz-layout>
  `,
  styleUrl: './menu.component.scss'
})
export class MenuComponent {

}
