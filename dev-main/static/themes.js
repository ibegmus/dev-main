(() => {
  'use strict';
  const choices={forest:'Лес',field:'Поле',plain:'Без фото'};
  let selected='forest';
  try{const saved=localStorage.getItem('terramind-theme');if(Object.hasOwn(choices,saved))selected=saved;}catch{}
  document.documentElement.dataset.scene=selected;
  document.addEventListener('DOMContentLoaded',()=>{
    for(const target of [document.querySelector('.mobile-brandbar'),document.querySelector('.sidebar')]){
      if(!target)continue;
      const label=document.createElement('label');label.className='theme-picker';label.textContent='Фон ';
      const select=document.createElement('select');select.setAttribute('aria-label','Оформление сайта');
      for(const [value,text] of Object.entries(choices)){const option=new Option(text,value);select.add(option);}select.value=selected;
      select.onchange=()=>{selected=select.value;document.documentElement.dataset.scene=selected;document.querySelectorAll('.theme-picker select').forEach(el=>el.value=selected);try{localStorage.setItem('terramind-theme',selected);}catch{}};
      label.append(select);target.append(label);
    }
  });
})();
