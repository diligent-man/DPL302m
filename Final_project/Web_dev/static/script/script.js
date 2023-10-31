function toggleMenu() {
  const menu = document.querySelector(".menu-links");
  const icon = document.querySelector(".hamburger_icon");
  menu.classList.toggle("open");
  icon.classList.toggle("open");
}

let input_area = document.getElementById("input_text")
input_area.addEventListener("input", ( event ) => {
    let count_word = document.getElementById("input_chars")
    let alr = document.getElementById("alert")
    let number_of_words = input_area.value.length
    if (number_of_words > 10){
      alr.style["display"] = "block"
      input_area.value = input_area.value.substring(0, number_of_words - 1)
    }
    else{
      count_word.textContent = number_of_words
      alr.style["display"] = "none"
    }
})
function changeLanguage() {
  var selectedLanguage = document.getElementById("language").value;
  var input_text = document.getElementById("input_text");

  if (selectedLanguage === "English") {
      input_text.setAttribute("placeholder", "Write something here...");
  } else if (selectedLanguage === "Vietnamese") {
      input_text.setAttribute("placeholder", "Viết gì đó ở đây...");
  }

}
    function changeToCurrentLanguage(){
        let cur_lang = document.getElementById("cached").textContent
        cur_lang = cur_lang.trim()
        let lang_selection = document.getElementById("language")
        if (cur_lang === 'English'){
            lang_selection.select = 'English'
        }
        else {
            lang_selection.value = 'Vietnamese'
        }
        console.log(lang_selection.value)
    }
