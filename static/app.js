var buttonNet = document.getElementById("net");




buttonNet.onclick = function() {
    
   
    
	  var xhr = new XMLHttpRequest();
    xhr.onreadystatechange = function() {
        if (xhr.readyState == 4 && xhr.status == 200) {
            // alert(xhr.responseText);
        }
    }


   
   
    xhr.open("POST", "/net");
    xhr.setRequestHeader("Content-Type", "application/json;charset=UTF-8");
    xhr.send(JSON.stringify({ status: "true" }));

};

