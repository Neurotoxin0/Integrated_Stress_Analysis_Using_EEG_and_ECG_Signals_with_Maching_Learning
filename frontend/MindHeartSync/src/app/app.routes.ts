/*Author: Xilai Wang*/
import { Routes } from '@angular/router';
import { MenuComponent } from './menu/menu.component';
import { HardwareComponent } from './hardware/hardware.component';
import { ModelSelectionComponent } from './model-selection/model-selection.component';
import { UserinputComponent } from './userinput/userinput.component';
import { NzNotificationModule } from 'ng-zorro-antd/notification';

// This is the root route configuration.
export const routes: Routes = [
  { path: '', pathMatch: 'full', redirectTo: 'welcome' },  // If you enter localhost:8080/, it will redirect to the component specified by the welcome route.
  { path: 'welcome', loadChildren: () => import('./pages/welcome/welcome.routes').then(m => m.WELCOME_ROUTES) },
  { path : 'menu', component:MenuComponent, title:'menu'},
  { path: 'hardware', component: HardwareComponent, title:'hardware-connection'},
  { path: 'model-selection', component: ModelSelectionComponent, title:'model-selection'},
  { path: 'userinput/:chosen_model', component: UserinputComponent, title:'input-page', providers:[NzNotificationModule]},
];
