/*Author: Xilai Wang*/
/*the root component of the whole Angular application. */
import { Component,CUSTOM_ELEMENTS_SCHEMA } from '@angular/core';
import { CommonModule } from '@angular/common';
import { RouterOutlet } from '@angular/router';
import { NzIconModule } from 'ng-zorro-antd/icon';
import { NzLayoutModule } from 'ng-zorro-antd/layout';
import { NzMenuModule } from 'ng-zorro-antd/menu';
import { NzImageModule } from 'ng-zorro-antd/image';
import { NzAvatarModule } from 'ng-zorro-antd/avatar';
import { NzCarouselModule } from 'ng-zorro-antd/carousel';
import { NzButtonModule } from 'ng-zorro-antd/button';
import { NzTypographyModule } from 'ng-zorro-antd/typography';
import { RouterModule } from '@angular/router';
import { provideRouter } from '@angular/router';
import { NzNotificationModule } from 'ng-zorro-antd/notification';
@Component({
  selector: 'app-root',
  standalone: true,
  imports: [NzNotificationModule, CommonModule, RouterOutlet, NzIconModule, NzLayoutModule, NzMenuModule, NzImageModule,NzAvatarModule,NzCarouselModule,NzButtonModule,NzTypographyModule,
    RouterModule],
    
  templateUrl:'./app.component.html',
  styleUrls: ['./app.component.scss']
})
export class AppComponent {
  isCollapsed = false;
  

      
}
