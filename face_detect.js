// face detect

import xapi from 'xapi';

const zeroPeople = 0;
const myUrl = 'https://www.google.com';


// Process updated PEOPLE COUNT data
function checkPeopleCount(amount) {
   console.log('DEBUG - Detected: ' + amount);
   if (amount > zeroPeople) {
      xapi.command('UserInterface WebView Display', {
        Title: 'Welcome Page',
        Url: myUrl
      });

   }else if(amount == zeroPeople){
     xapi.command('UserInterface WebView Clear');
   }
}


// Get updated PEOPLE COUNT value
xapi.status.on('RoomAnalytics PeopleCount Current', (numberofpeople) => checkPeopleCount(numberofpeople)); 