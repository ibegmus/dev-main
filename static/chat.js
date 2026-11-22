(() => {
  'use strict';
  const $ = id => document.getElementById(id);
  const chatWindow=$('chat-window'), chatInput=$('chat-input'), sendBtn=$('send-btn'),
    newChatBtn=$('new-chat'), chatNotice=$('chat-notice'), fileUpload=$('file-upload'),
    photoPreview=$('photo-preview'), photoThumbnail=$('photo-thumbnail'), photoName=$('photo-name'),
    uploadError=$('upload-error'), authDialog=$('auth-dialog');
  const conversationKey='terramind-conversation', pendingJobKey='terramind-pending-job', ownerKey='terramind-conversation-owner';
  let user=null, authReady=false, authBusy=false, authMode='login', accountEpoch=0;
  let conversationId=null, conversationEpoch=0, isSending=false, isReadingPhoto=false, selectedPhoto=null;
  let newChatUntil=0, chats=[], listVersion=0, nextBefore=null, paging=false, checkingIdentity=false;
  const delay=ms=>new Promise(resolve=>setTimeout(resolve,ms));
  const owner=()=>user ? `account:${user.id}` : 'guest';
  const stored=key=>{try{return sessionStorage.getItem(key);}catch{return null;}};
  function store(key,value){try{if(value)sessionStorage.setItem(key,value);else sessionStorage.removeItem(key);}catch{}}
  function saveConversation(id){conversationId=id||null;store(conversationKey,conversationId);store(ownerKey,conversationId?owner():null);renderChatList();}
  function savePendingJob(id){store(pendingJobKey,id);}
  async function request(path,{method='GET',body,timeout=15000}={}){
    const options={method,credentials:'same-origin',cache:'no-store',headers:{'Accept':'application/json'}};
    if(method!=='GET')options.headers['X-TerraMind-Request']='1';
    if(body!==undefined){options.headers['Content-Type']='application/json';options.body=JSON.stringify(body);}
    if(timeout)options.signal=AbortSignal.timeout(timeout);
    const response=await fetch(path,options);
    let data={};try{if(response.status!==204)data=await response.json();}catch{}
    if(!response.ok){const error=new Error(typeof data.detail==='string'?data.detail:'Не удалось выполнить запрос. Попробуйте ещё раз.');error.status=response.status;error.retry=response.headers.get('Retry-After');throw error;}
    return data;
  }
  function controls(){sendBtn.setAttribute('aria-label',isSending?'Ожидаем ответ':'Отправить сообщение');sendBtn.title=isSending?'Ожидаем ответ':'Отправить сообщение';const busy=isSending||isReadingPhoto||authBusy||!authReady;sendBtn.disabled=busy;sendBtn.textContent=isSending?'Ожидаем ответ…':'Отправить';chatInput.disabled=isSending||authBusy||!authReady;
    for(const id of ['file-upload','attach-photo','remove-photo'])$(id).disabled=busy;
    document.querySelectorAll('.quick-btn').forEach(b=>b.disabled=busy);
    newChatBtn.disabled=authBusy||!authReady||Date.now()<newChatUntil;
    for(const id of ['open-auth','logout-btn','open-account-delete','auth-submit','auth-login-tab','auth-register-tab','close-auth','account-delete-submit','close-account-delete','open-security','close-security','change-password-submit','toggle-auth-password'])$(id).disabled=authBusy;
  }
  function accountView(){
    $('feedback-admin-dialog').close();$('feedback-admin-list').replaceChildren();$('open-feedback-admin').hidden=true;$('quota-status').textContent='';
    refreshUsage();
    $('open-auth').hidden=!!user;$('logout-btn').hidden=!user;$('open-account-delete').hidden=!user;$('open-security').hidden=!user;$('saved-chats').hidden=!user;
    $('account-label').textContent=user?user.name:'Гостевой чат';
    $('account-subtitle').textContent=user?'Ваши беседы сохраняются в аккаунте.':'Войдите, чтобы сохранять беседы и возвращаться к ним.';
    $('memory-note').textContent=user?'Беседы сохраняются до удаления. ИИ учитывает недавние сообщения выбранного чата и описание последнего фото. «Новый чат» сохраняет прежнюю беседу в списке.':'ИИ учитывает до 8 последних обменов и описание фото. Гостевая беседа хранится 30 минут без новых сообщений. В аккаунте беседы сохраняются до удаления.';
    controls();
  }
  function clearPhoto(){selectedPhoto=null;fileUpload.value='';photoPreview.hidden=true;photoThumbnail.removeAttribute('src');photoName.textContent='';}
  function hideQueue(){$('queue-banner').classList.remove('active');}
  function emptyChat(message='Здравствуйте! Что вы хотите узнать о почве?'){
    ++conversationEpoch;saveConversation(null);savePendingJob(null);clearPhoto();isSending=false;isReadingPhoto=false;paging=false;nextBefore=null;$('older-messages').hidden=true;chatInput.value='';uploadError.textContent='';chatWindow.replaceChildren();addMessage(message,'ai');hideQueue();controls();
  }
  function parseMarkdown(text){return text.replace(/\*\*(.*?)\*\*/g,'$1').replace(/\*(.*?)\*/g,'$1').replace(/__(.*?)__/g,'$1').replace(/_(.*?)_/g,'$1');}
  function renderAnswer(container,text,copyable=false){container.classList.remove('is-waiting');container.removeAttribute('data-wait-state');container.replaceChildren();for(const part of String(text||'').split(/(https?:\/\/[^\s<>]+)/g)){
    if(/^https?:\/\//.test(part)){const address=part.replace(/[.,;!?]+$/,'');try{const url=new URL(address);if(!['http:','https:'].includes(url.protocol))throw Error();const a=document.createElement('a');a.className='source-link';a.href=url.href;a.textContent=address;a.target='_blank';a.rel='noopener noreferrer';container.append(a,document.createTextNode(part.slice(address.length)));continue;}catch{}}
    container.append(document.createTextNode(parseMarkdown(part)));
  }
    if (copyable && String(text || '').trim()) addCopyControl(container, String(text));
  }
  function addCopyControl(container, text) {
    const tools = document.createElement('div');
    tools.className = 'answer-tools';
    const copy = document.createElement('button');
    copy.type = 'button';
    copy.className = 'copy-answer';
    copy.textContent = 'Скопировать ответ';
    copy.title = 'Скопировать ответ вместе с источниками';
    const status = document.createElement('span');
    status.className = 'copy-status';
    status.setAttribute('role', 'status');
    copy.addEventListener('click', async () => {
      copy.disabled = true;
      status.textContent = '';
      try {
        if (!navigator.clipboard?.writeText) throw new Error('Clipboard unavailable');
        await navigator.clipboard.writeText(text);
        status.textContent = 'Скопировано';
      } catch (_) {
        status.textContent = 'Не удалось скопировать. Выделите текст вручную.';
      } finally {
        copy.disabled = false;
      }
    });
    tools.append(copy, status);
    container.append(tools);
  }
  function messageNode(text,sender,photo,copyable=false){const div=document.createElement('div');div.className=`message ${sender}`;if(sender==='ai')renderAnswer(div,text,copyable);else div.textContent=text;if(photo){const image=document.createElement('img');image.src=photo.dataUrl;image.alt='Фотография, отправленная нейросети';div.append(image);}return div;}
  function addMessage(text,sender,photo=null){const div=messageNode(text,sender,photo);chatWindow.append(div);chatWindow.scrollTop=chatWindow.scrollHeight;return div;}
  const soilVisual=`<svg viewBox="0 0 110 144" aria-hidden="true"><defs><clipPath id="soil-profile-clip"><rect x="22" y="32" width="66" height="104" rx="10"/></clipPath></defs><g clip-path="url(#soil-profile-clip)"><path fill="#455841" d="M22 32h66v25H22z"/><path fill="#8b7459" d="M22 56q30 9 66-2v30H22z"/><path fill="#b59b76" d="M22 82q27-7 66 3v29H22z"/><path fill="#d6c5a5" d="M22 111q36 7 66-3v28H22z"/><g fill="#e7dcc7"><circle class="soil-particle" cx="34" cy="68" r="2"/><circle class="soil-particle" cx="65" cy="75" r="1.5"/><circle class="soil-particle" cx="77" cy="94" r="2"/><circle class="soil-particle" cx="42" cy="101" r="2.5"/><circle class="soil-particle" cx="63" cy="122" r="3"/><circle class="soil-particle" cx="32" cy="124" r="1.5"/></g><g class="soil-scan"><rect x="17" y="34" width="76" height="8" fill="#cfedb2" opacity=".3"/><path d="M17 38h76" stroke="#d9f2bb" stroke-width="2"/></g></g><rect x="22" y="32" width="66" height="104" rx="10" fill="none" stroke="#456b4d" stroke-opacity=".22"/><g class="soil-sprout"><path d="M55 34V17" stroke="#497d4f" stroke-width="2"/><path d="M55 26C42 25 39 16 41 12c10-1 15 5 14 14" fill="#7da567"/><path d="M55 20C55 9 62 5 70 7c0 9-6 14-15 13" fill="#a2c181"/></g><path d="M15 51h-5m5 30h-5m5 30h-5M95 51h5m-5 30h5m-5 30h5" stroke="#9ba88e"/></svg>`;
  function waiting(div, text, state = 'sending') {
    const follow = chatWindow.scrollHeight - chatWindow.scrollTop - chatWindow.clientHeight < 80;
    if (!div.querySelector('.soil-wait')) {
      div.replaceChildren();
      div.classList.add('is-waiting');
      const row = document.createElement('div');
      row.className = 'soil-wait';
      row.innerHTML = `<div class="soil-illustration">${soilVisual}</div>
        <div class="soil-wait-copy"><p class="soil-wait-kicker">ИИ-почвовед · TerraMind</p>
        <h3 class="soil-wait-text"></h3><p class="soil-wait-description"></p></div>
        <div class="soil-wait-method" aria-label="Как готовится ответ">
          <span>Вопрос</span><span aria-hidden="true">→</span><span>Материалы</span><span aria-hidden="true">→</span><span>Объяснение</span>
        </div>`;
      div.append(row);
    }
    div.dataset.waitState = state;
    const title = div.querySelector('.soil-wait-text');
    const description = div.querySelector('.soil-wait-description');
    const detail = {
      sending: 'Устанавливаем связь и передаём ваш вопрос.',
      preparing: 'Подготавливаем модели к работе. Ваш вопрос уже принят.',
      queued: 'Начнём, когда ИИ закончит предыдущие запросы. Место в очереди сохраняется.',
      processing: 'Анализируем вопрос, сопоставляем материалы и формулируем объяснение.'
    };
    if (title.textContent !== text) title.textContent = text;
    const nextDescription = detail[state] || '';
    if (description.textContent !== nextDescription) description.textContent = nextDescription;
    div.querySelector('.soil-wait-method').hidden = state !== 'processing';
    if (follow && div.isConnected) chatWindow.scrollTop = chatWindow.scrollHeight;
  }

  function updateQueue(job, div) {
    $('queue-banner').classList.add('active');
    const preparing = job.preparing === true;
    const processing = job.status === 'processing';
    const state = preparing ? 'preparing' : processing ? 'processing' : 'queued';
    $('queue-title').textContent = preparing ? 'ИИ готовится к работе'
      : processing ? 'ИИ обрабатывает ваш запрос' : 'Ваш запрос в очереди';
    $('queue-count').textContent = job.ahead ?? 0;
    $('queue-details').hidden = processing || preparing;
    $('queue-timer').textContent = preparing ? 'Подготовка' : processing ? 'Обработка' : 'В очереди';
    waiting(div, preparing ? 'Подготавливаем ИИ'
      : processing ? 'Собираем ответ для вас'
      : `Вопрос в очереди · перед вами ${job.ahead ?? 0}`, state);
  }

  function lostAccount(){++accountEpoch;resetAccountForms();user=null;chats=[];emptyChat();accountView();chatNotice.textContent='Вход в аккаунт истёк или изменился. Войдите снова, чтобы продолжить сохранённую беседу.';}
  function chatError(error,div){div.classList.remove('is-waiting');div.removeAttribute('data-wait-state');if(error.status===401&&user){lostAccount();return;}div.textContent=error.message||'Нет связи с сервером. Попробуйте ещё раз.';if(error.status===429&&error.retry)div.textContent+=` Повторите через ${error.retry} с.`;if(error.status===410||error.status===404){saveConversation(null);chatNotice.textContent='Эта беседа недоступна. Следующий вопрос начнёт новый чат.';}}
  async function refreshUsage(force=false){
    const identity=accountEpoch, cid=conversationId;
    try{
      const state=await request('/account/usage');
      if(identity!==accountEpoch||cid!==conversationId)return;
      $('quota-status').textContent=state.admin?'Администратор · без дневного лимита':Number.isInteger(state.remaining)?`Сегодня осталось ${state.remaining} из ${state.limit} запросов · обновление в 00:00 ЕКБ`:'';
      $('open-feedback-admin').hidden=!state.admin;
      if(state.bonus_available && cid && (force || chatWindow.querySelectorAll('.answer-tools').length>=3))offerFeedback(cid);
    }catch{ /* Quota availability never blocks displaying a completed answer. */ }
  }
  function offerFeedback(cid){
    const key=`feedback-dismissed:${owner()}:${cid}`;
    if(!user||stored(key)||chatWindow.querySelector('.feedback-card'))return;
    const identity=accountEpoch;
    const card=document.createElement('form');card.className='feedback-card';
    card.innerHTML='<h3>Как вам беседа с ИИ-почвоведом?</h3><details><summary>Оценить беседу · +5 запросов</summary><fieldset><legend>Ваша оценка</legend><div class="feedback-stars">'+[1,2,3,4,5].map(n=>`<label><input type="radio" name="stars" value="${n}" required><span aria-hidden="true">★</span><span class="feedback-rating-label">${n} из 5</span></label>`).join('')+'</div></fieldset><label class="feedback-comment-label">Что было полезно, а что улучшить?<textarea name="comment" minlength="10" maxlength="2000" rows="3" required placeholder="Ваш отзыв поможет улучшить ответы"></textarea></label><p>За первый отзыв с комментарием — ещё 5 запросов сегодня. Оценка может быть любой. Отзыв видят только администраторы.</p><div class="feedback-actions"><button type="submit" class="account-primary">Отправить отзыв</button></div></details><button type="button" class="account-text">Не сейчас</button><p role="status" class="feedback-status"></p>';
    card.querySelector('details').addEventListener('toggle',()=>{if(card.querySelector('details')?.open)card.scrollIntoView({block:'nearest'});});
    card.querySelector('button[type=button]').onclick=()=>{store(key,'1');card.remove();};
    card.onsubmit=async event=>{
      event.preventDefault();const button=card.querySelector('button[type=submit]'),status=card.querySelector('.feedback-status');button.disabled=true;
      try{await request('/feedback',{method:'POST',body:{conversation_id:cid,stars:Number(new FormData(card).get('stars')),comment:new FormData(card).get('comment')}});
        if(identity!==accountEpoch||cid!==conversationId)return;
        store(key,'1');card.replaceChildren();const thanks=document.createElement('p');thanks.textContent='Спасибо за отзыв! Добавлено 5 запросов на сегодня.';card.append(thanks);refreshUsage();
      }catch(error){if(card.isConnected){status.textContent=error.message;button.disabled=false;}}
    };
    const atBottom=chatWindow.scrollHeight-chatWindow.scrollTop-chatWindow.clientHeight<100;
    chatWindow.append(card);if(atBottom)chatWindow.scrollTop=chatWindow.scrollHeight;
  }
  $('open-feedback-admin').addEventListener('click',async()=>{
    const identity=accountEpoch,dialog=$('feedback-admin-dialog'),list=$('feedback-admin-list');dialog.showModal();list.textContent='Загружаем отзывы…';
    try{const data=await request('/admin/feedback');if(identity!==accountEpoch)return;list.replaceChildren();for(const item of data.feedback){const article=document.createElement('article'),heading=document.createElement('strong'),comment=document.createElement('p');heading.textContent=`${item.stars} / 5 · ${item.created_at}`;comment.textContent=item.comment;article.append(heading,comment);list.append(article);}if(!data.feedback.length)list.textContent='Отзывов пока нет.';}catch(error){if(identity===accountEpoch)list.textContent=error.message;}
  });
  $('close-feedback-admin').addEventListener('click',()=>$('feedback-admin-dialog').close());

  function renderChatList(){const list=$('saved-chat-list');list.replaceChildren();if(!user)return;for(const chat of chats){const item=document.createElement('div');item.className='saved-chat-item'+(chat.conversation_id===conversationId?' active':'');const open=document.createElement('button');open.type='button';open.className='saved-chat-open';open.dataset.conversationId=chat.conversation_id;const title=document.createElement('strong');title.textContent=chat.title||'Новая беседа';const date=document.createElement('small');const parsed=new Date(chat.updated_at);date.textContent=Number.isNaN(parsed.getTime())?'Сохранено':parsed.toLocaleDateString('ru-RU',{day:'numeric',month:'short'});open.append(title,date);open.addEventListener('click',()=>openChat(chat.conversation_id));const remove=document.createElement('button');remove.type='button';remove.className='saved-chat-delete';remove.textContent='×';remove.setAttribute('aria-label',`Удалить беседу «${chat.title||'Новая беседа'}»`);remove.addEventListener('click',()=>removeChat(chat));item.append(open,remove);list.append(item);}}
  async function refreshChats(){if(!user)return;const epoch=accountEpoch,version=++listVersion;$('saved-chat-status').textContent='Загружаем беседы…';try{const data=await request('/conversations');if(epoch!==accountEpoch||version!==listVersion)return;chats=data.conversations||[];renderChatList();$('saved-chat-status').textContent=chats.length?`${chats.length} из 50 бесед · нажмите, чтобы продолжить`:'Здесь появится ваша первая сохранённая беседа.';}catch(error){if(epoch!==accountEpoch||version!==listVersion)return;if(error.status===401){lostAccount();return;}$('saved-chat-status').textContent='Не удалось загрузить список. Нажмите «Обновить».';}}
  async function waitForAnswer(jobId,div,epoch){let errors=0;while(epoch===conversationEpoch){let job;try{job=await request(`/ask/jobs/${encodeURIComponent(jobId)}`);}catch(error){if(epoch!==conversationEpoch)return;if(error.status&&error.status<500)throw error;waiting(div,'Временно нет связи. Продолжаем ждать ваш ответ…');await delay(Math.min(++errors*2000,10000));continue;}if(epoch!==conversationEpoch)return;errors=0;if(job.conversation_id)saveConversation(job.conversation_id);if(job.status==='done'){renderAnswer(div,job.answer,true);refreshUsage();return;}if(job.status==='failed')throw Error(job.error||'Не удалось получить ответ. Попробуйте ещё раз.');updateQueue(job,div);await delay(1000);}}
  async function sendMessage(question){question=question.trim();if(isSending||isReadingPhoto||authBusy||!authReady||(!question&&!selectedPhoto))return;if(question.length>4000){uploadError.textContent='Вопрос слишком длинный. Максимум — 4000 символов.';return;}const epoch=conversationEpoch,identity=accountEpoch,wasGuest=!user,oldId=conversationId;isSending=true;controls();uploadError.textContent='';chatNotice.textContent='';const photo=selectedPhoto;addMessage(question||'Что изображено на этом фото?','user',photo);const div=addMessage('','ai');waiting(div,'Отправляем запрос…');try{const job=await request('/ask/jobs',{method:'POST',body:{text:question,image:photo?photo.base64:null,conversation_id:oldId},timeout:0});if(epoch!==conversationEpoch||identity!==accountEpoch){if(wasGuest&&job.conversation_id)request(`/conversations/${encodeURIComponent(job.conversation_id)}`,{method:'DELETE'}).catch(()=>{});if(!wasGuest&&identity===accountEpoch)refreshChats();return;}saveConversation(job.conversation_id);savePendingJob(job.job_id);chatInput.value='';clearPhoto();updateQueue(job,div);if(user)refreshChats();await waitForAnswer(job.job_id,div,epoch);}catch(error){if(epoch===conversationEpoch){chatError(error,div);if(error.status===429)refreshUsage(true);}}finally{if(epoch===conversationEpoch){savePendingJob(null);hideQueue();isSending=false;controls();chatInput.focus();chatWindow.scrollTop=chatWindow.scrollHeight;if(user)refreshChats();}}}
  function addTurns(turns,prepend=false){const fragment=document.createDocumentFragment();for(const turn of turns){fragment.append(messageNode(turn.question+(turn.has_image?' [с фото]':''),'user'),messageNode(turn.answer,'ai',null,true));}if(prepend)chatWindow.prepend(fragment);else chatWindow.append(fragment);}
  async function openChat(id){if(authBusy||!authReady)return;const epoch=++conversationEpoch;saveConversation(id);savePendingJob(null);clearPhoto();chatInput.value='';isSending=true;isReadingPhoto=false;nextBefore=null;$('older-messages').hidden=true;hideQueue();controls();chatWindow.replaceChildren();const div=addMessage('Загружаем беседу…','ai');try{const data=await request(`/conversations/${encodeURIComponent(id)}`);if(epoch!==conversationEpoch)return;chatWindow.replaceChildren();addTurns(data.turns||[]);if(!chatWindow.children.length)addMessage('Здравствуйте! Что вы хотите узнать о почве?','ai');nextBefore=data.has_more?data.next_before:null;$('older-messages').hidden=!nextBefore;chatNotice.textContent='';chatWindow.scrollTop=chatWindow.scrollHeight;if(data.active_job_id){savePendingJob(data.active_job_id);const waitingDiv=addMessage('','ai');waiting(waitingDiv,'Продолжаем ждать ответ…');await waitForAnswer(data.active_job_id,waitingDiv,epoch);}}catch(error){if(epoch!==conversationEpoch)return;const target=div.isConnected?div:addMessage('','ai');chatError(error,target);}finally{if(epoch===conversationEpoch){savePendingJob(null);isSending=false;hideQueue();controls();refreshUsage();if(user)refreshChats();}}}
  async function removeChat(chat){if(authBusy||!confirm(`Удалить беседу «${chat.title||'Новая беседа'}»? Это действие нельзя отменить.`))return;const identity=accountEpoch;try{await request(`/conversations/${encodeURIComponent(chat.conversation_id)}`,{method:'DELETE'});if(identity!==accountEpoch)return;if(conversationId===chat.conversation_id){emptyChat();chatNotice.textContent='Беседа удалена.';}await refreshChats();}catch(error){if(identity===accountEpoch){if(error.status===401)lostAccount();else chatNotice.textContent=error.message;}}}
  newChatBtn.addEventListener('click',async()=>{if(Date.now()<newChatUntil||authBusy||!authReady)return;newChatUntil=Date.now()+2000;const previous=conversationId,wasGuest=!user;emptyChat();const epoch=conversationEpoch;chatNotice.textContent=wasGuest?'Новая беседа. Предыдущая память очищается.':'Новая беседа. Прежний чат сохранён в списке.';setTimeout(controls,2050);chatInput.focus();if(previous&&wasGuest){try{await request(`/conversations/${encodeURIComponent(previous)}`,{method:'DELETE'});}catch(error){if(epoch===conversationEpoch)chatNotice.textContent='Новая беседа начата. Очистку прежней памяти подтвердить не удалось; она истечёт автоматически.';}}if(user)refreshChats();});
  $('older-messages').addEventListener('click',async()=>{if(!nextBefore||paging||!conversationId)return;const epoch=conversationEpoch,id=conversationId;paging=true;$('older-messages').disabled=true;try{const data=await request(`/conversations/${encodeURIComponent(id)}?before=${encodeURIComponent(nextBefore)}`);if(epoch!==conversationEpoch)return;const height=chatWindow.scrollHeight,position=chatWindow.scrollTop;addTurns(data.turns||[],true);chatWindow.scrollTop=position+chatWindow.scrollHeight-height;nextBefore=data.has_more?data.next_before:null;$('older-messages').hidden=!nextBefore;}catch(error){if(epoch===conversationEpoch)chatNotice.textContent=error.message;}finally{if(epoch===conversationEpoch){paging=false;$('older-messages').disabled=false;}}});
  $('refresh-chats').addEventListener('click',refreshChats);
  function setAuthMode(mode){authMode=mode;const register=mode==='register';$('auth-title').textContent=register?'Создайте аккаунт':'С возвращением';$('auth-description').textContent=register?'Сохраняйте беседы и продолжайте их в любое время.':'Войдите, чтобы продолжить сохранённую беседу.';$('auth-name-label').hidden=!register;$('auth-name').required=register;$('auth-terms-label').hidden=!register;$('auth-terms').required=register;$('password-hint').hidden=!register;$('auth-password').minLength=register?12:1;$('auth-password').type='password';$('toggle-auth-password').textContent='Показать';$('toggle-auth-password').setAttribute('aria-pressed','false');$('auth-password').autocomplete=register?'new-password':'current-password';$('auth-submit').textContent=register?'Создать аккаунт':'Войти';$('auth-login-tab').setAttribute('aria-pressed',String(!register));$('auth-register-tab').setAttribute('aria-pressed',String(register));$('auth-error').textContent='';}
  $('open-auth').addEventListener('click',()=>{setAuthMode('login');authDialog.showModal();$('auth-email').focus();});$('close-auth').addEventListener('click',()=>authDialog.close());$('auth-login-tab').addEventListener('click',()=>setAuthMode('login'));$('auth-register-tab').addEventListener('click',()=>setAuthMode('register'));
  for(const dialog of [authDialog,$('account-delete-dialog'),$('security-dialog')])dialog.addEventListener('cancel',event=>{if(authBusy)event.preventDefault();});
  $('auth-form').addEventListener('submit',async event=>{event.preventDefault();if(authBusy)return;authBusy=true;controls();$('auth-error').textContent='';const register=authMode==='register';const body={email:$('auth-email').value.trim(),password:$('auth-password').value};if(register){body.name=$('auth-name').value.trim();body.terms_accepted=$('auth-terms').checked;}try{const data=await request(register?'/register':'/login',{method:'POST',body});++accountEpoch;user=data.user;authReady=true;chats=[];emptyChat();accountView();authDialog.close();$('auth-password').value='';$('auth-terms').checked=false;chatNotice.textContent=register?'Аккаунт создан. Новые беседы будут сохраняться автоматически.':'Вы вошли. Выберите сохранённую беседу или начните новую.';await refreshChats();}catch(error){$('auth-error').textContent=error.message+(error.status===429&&error.retry?` Повторите через ${error.retry} с.`:'');}finally{authBusy=false;controls();}});
  $('logout-btn').addEventListener('click',async()=>{if(authBusy)return;authBusy=true;controls();try{await request('/logout',{method:'POST'});++accountEpoch;resetAccountForms();user=null;chats=[];emptyChat();accountView();chatNotice.textContent='Вы вышли. Сохранённые беседы доступны после входа.';}catch(error){chatNotice.textContent='Не удалось выйти из аккаунта. Проверьте соединение и попробуйте снова.';}finally{authBusy=false;controls();}});
  function resetAccountForms() {
    $('password-form').reset();
    $('security-dialog').close();
    $('account-delete-form').reset();
    $('account-delete-dialog').close();
  }
  $('toggle-auth-password').addEventListener('click', () => {
    const visible = $('auth-password').type === 'password';
    $('auth-password').type = visible ? 'text' : 'password';
    $('toggle-auth-password').textContent = visible ? 'Скрыть' : 'Показать';
    $('toggle-auth-password').setAttribute('aria-pressed', String(visible));
  });
  $('open-security').addEventListener('click', () => {
    $('password-form').reset();
    $('password-error').textContent = '';
    $('security-email').textContent = user?.email || '';
    $('security-username').value = user?.email || '';
    $('security-dialog').showModal();
  });
  $('close-security').addEventListener('click', () => $('security-dialog').close());
  $('password-form').addEventListener('submit', async event => {
    event.preventDefault();
    if (authBusy) return;
    $('password-error').textContent = '';
    if ($('new-password').value !== $('confirm-password').value) {
      $('password-error').textContent = 'Новые пароли не совпадают.';
      return;
    }
    authBusy = true; controls();
    try {
      await request('/account/password', {method:'POST', body:{current_password:$('current-password').value, new_password:$('new-password').value}});
      $('password-form').reset();
      $('security-dialog').close();
      chatNotice.textContent = 'Пароль изменён. Остальные сеансы завершены; ваши чаты сохранены.';
    } catch (error) {
      if (error.status === 401) lostAccount();
      else $('password-error').textContent = error.message;
    } finally { authBusy = false; controls(); }
  });
  $('open-account-delete').addEventListener('click',()=>{$('security-dialog').close();$('account-delete-password').value='';$('account-delete-error').textContent='';$('account-delete-dialog').showModal();$('account-delete-password').focus();});$('close-account-delete').addEventListener('click',()=>$('account-delete-dialog').close());
  $('account-delete-form').addEventListener('submit',async event=>{event.preventDefault();if(authBusy)return;authBusy=true;controls();try{await request('/account',{method:'DELETE',body:{password:$('account-delete-password').value}});++accountEpoch;resetAccountForms();user=null;chats=[];emptyChat();accountView();$('account-delete-dialog').close();$('account-delete-password').value='';chatNotice.textContent='Аккаунт и сохранённые беседы удалены.';}catch(error){$('account-delete-error').textContent=error.message;}finally{authBusy=false;controls();}});
  async function checkIdentity(initial=false){if(checkingIdentity||authBusy)return;checkingIdentity=true;const identity=accountEpoch;try{let next;try{next=(await request('/me')).user;}catch(error){if(error.status!==401)throw error;next=null;}if(identity!==accountEpoch)return;const changed=String(user?.id||'')!==String(next?.id||'');user=next;authReady=true;if(changed&&!initial){++accountEpoch;resetAccountForms();chats=[];emptyChat();chatNotice.textContent='Учётная запись изменилась. Выберите беседу для продолжения.';}accountView();if(user)await refreshChats();if(initial){const id=stored(conversationKey),savedOwner=stored(ownerKey)||'guest',job=stored(pendingJobKey);if(savedOwner!==owner()){saveConversation(null);savePendingJob(null);}else if(id){await openChat(id);}else if(job&&!user){const epoch=conversationEpoch;isSending=true;controls();const div=addMessage('','ai');waiting(div,'Восстанавливаем ожидание ответа…');try{await waitForAnswer(job,div,epoch);}catch(error){if(epoch===conversationEpoch){chatError(error,div);if(error.status===429)refreshUsage(true);}}finally{if(epoch===conversationEpoch){savePendingJob(null);isSending=false;hideQueue();controls();}}}}}catch(error){if(identity===accountEpoch){chatNotice.textContent='Не удалось проверить вход. Обновите страницу или попробуйте войти снова.';}}finally{checkingIdentity=false;controls();document.documentElement.dataset.chatReady=String(authReady);}}
  window.addEventListener('focus',()=>checkIdentity());document.addEventListener('visibilitychange',()=>{if(!document.hidden)checkIdentity();});
  $('burger-menu').addEventListener('click',()=>{const visible=$('sidebar').classList.toggle('visible');$('burger-menu').setAttribute('aria-expanded',String(visible));$('overlay').classList.toggle('active');});$('overlay').addEventListener('click',()=>{$('sidebar').classList.remove('visible');$('burger-menu').setAttribute('aria-expanded','false');$('overlay').classList.remove('active');});
  $('attach-photo').addEventListener('click',()=>fileUpload.click());$('remove-photo').addEventListener('click',clearPhoto);
  fileUpload.addEventListener('change',async()=>{const file=fileUpload.files[0];if(!file||isSending)return;clearPhoto();uploadError.textContent='';if(!['image/jpeg','image/png','image/webp'].includes(file.type)){uploadError.textContent='Выберите фотографию JPEG, PNG или WebP.';return;}if(file.size>5*1024*1024){uploadError.textContent='Фото слишком большое. Максимальный размер — 5 МБ.';return;}const epoch=conversationEpoch;isReadingPhoto=true;controls();try{const dataUrl=await new Promise((resolve,reject)=>{const reader=new FileReader();reader.onload=()=>resolve(reader.result);reader.onerror=()=>reject(Error('Не удалось прочитать фото.'));reader.readAsDataURL(file);});const image=new Image();image.src=dataUrl;await image.decode();if(epoch!==conversationEpoch)return;if(image.naturalWidth*image.naturalHeight>20000000)throw Error('Выберите фото до 20 мегапикселей.');selectedPhoto={dataUrl,base64:dataUrl.slice(dataUrl.indexOf(',')+1)};photoThumbnail.src=dataUrl;photoName.textContent=file.name;photoPreview.hidden=false;chatInput.focus();}catch(error){if(epoch===conversationEpoch){clearPhoto();uploadError.textContent=error.name==='EncodingError'?'Файл повреждён или не является фотографией.':error.message;}}finally{if(epoch===conversationEpoch){isReadingPhoto=false;controls();}}});
  sendBtn.addEventListener('click',()=>sendMessage(chatInput.value));chatInput.addEventListener('keydown',event=>{if(event.key==='Enter'&&!event.isComposing){event.preventDefault();sendMessage(chatInput.value);}});document.querySelectorAll('.quick-btn').forEach(button=>button.addEventListener('click',()=>sendMessage(button.dataset.q)));
  const welcomeScreen=$('welcome-screen');
  if(welcomeScreen){
    const welcomeSeen='terramind-welcome-seen';
    const canShowWelcome=!location.hash && ['/', '/index.html'].includes(location.pathname);
    let welcomeWasSeen=false;
    try{welcomeWasSeen=sessionStorage.getItem(welcomeSeen)==='1';}catch{}
    const forceWelcome=new URLSearchParams(location.search).has('welcome');
    if(canShowWelcome&&(!welcomeWasSeen||forceWelcome)){welcomeScreen.classList.remove('is-hidden');welcomeScreen.setAttribute('aria-hidden','false');}
    const closeWelcome=()=>{welcomeScreen.classList.add('is-hidden');welcomeScreen.setAttribute('aria-hidden','true');try{sessionStorage.setItem(welcomeSeen,'1');}catch{}};
    const reduceMotion=window.matchMedia&&window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    const flyLogoToHeader=(logo,target)=>{
      const from=logo.getBoundingClientRect(),to=target.getBoundingClientRect();
      if(!from.width||!to.width)return null;
      const clone=logo.cloneNode(true);
      clone.classList.add('logo-fly');clone.setAttribute('aria-hidden','true');
      Object.assign(clone.style,{left:`${from.left}px`,top:`${from.top}px`,width:`${from.width}px`,height:`${from.height}px`});
      document.body.appendChild(clone);
      const scale=Math.min(to.width/from.width,to.height/from.height,1);
      const dx=(to.left+to.width/2)-(from.left+from.width/2),dy=(to.top+to.height/2)-(from.top+from.height/2);
      requestAnimationFrame(()=>requestAnimationFrame(()=>{
        clone.style.transition='transform .62s cubic-bezier(.22,.8,.24,1), opacity .62s ease';
        clone.style.transform=`translate(${dx}px, ${dy}px) scale(${scale})`;
        clone.style.opacity='.94';
      }));
      return clone;
    };
    const startWelcome=()=>{
      const logo=document.querySelector('#welcome-logo .tm-logo');
      const targets=[...document.querySelectorAll('.mobile-brand .tm-logo, .sidebar .logo .tm-logo')];
      const target=targets.find(el=>el.getBoundingClientRect().width>0);
      if(reduceMotion||!logo||!target){closeWelcome();window.scrollTo({top:0,behavior:'auto'});return;}
      welcomeScreen.classList.add('is-leaving');
      window.setTimeout(()=>{
        const flying=flyLogoToHeader(logo,target);
        if(!flying){closeWelcome();return;}
        window.setTimeout(()=>{
          flying.remove();
          closeWelcome();
          target.classList.add('logo-landed');
          window.setTimeout(()=>target.classList.remove('logo-landed'),900);
          window.scrollTo({top:0,behavior:'smooth'});
        },760);
      },300);
    };
    const welcomeStart=$('welcome-start'),welcomeLogin=$('welcome-login');
    if(welcomeStart)welcomeStart.addEventListener('click',startWelcome);
    if(welcomeLogin)welcomeLogin.addEventListener('click',()=>{closeWelcome();$('open-auth').click();});
  }
  controls();checkIdentity(true);
})();
