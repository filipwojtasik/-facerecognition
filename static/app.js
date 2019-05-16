var buttonPic = document.getElementById("help");
var buttonStop = document.getElementById("stop");

buttonStop.disabled = true;

buttonPic.onclick = function() {
    buttonPic.disabled = true;
    buttonStop.disabled = false;
    
	  var xhr = new XMLHttpRequest();
    xhr.onreadystatechange = function() {
        if (xhr.readyState == 4 && xhr.status == 200) {
            // alert(xhr.responseText);
        }
    }


   
   
    xhr.open("POST", "/help");
    xhr.setRequestHeader("Content-Type", "application/json;charset=UTF-8");
    xhr.send(JSON.stringify({ status: "true" }));

};

buttonStop.onclick = function() {
    buttonPic.disabled = false;
    buttonStop.disabled = true;    

    var xhr = new XMLHttpRequest();
    xhr.onreadystatechange = function() {
        if (xhr.readyState == 4 && xhr.status == 200) {
            // alert(xhr.responseText);
        }
    }
   
    xhr.open("POST", "/help");
    xhr.setRequestHeader("Content-Type", "application/json;charset=UTF-8");
    xhr.send(JSON.stringify({ status: "false" }));

};

