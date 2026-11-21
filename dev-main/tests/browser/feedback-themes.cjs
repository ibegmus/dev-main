/* Local mocked API: feedback UI, quota notice, theme persistence and escaping. */
const {chromium}=require('playwright');
const assert=require('node:assert/strict');
const fs=require('node:fs'),path=require('node:path');
(async()=>{
 const browser=await chromium.launch({executablePath:'/usr/bin/chromium',args:['--no-sandbox']});
 for(const width of [320,390,1440]){
  const context=await browser.newContext({viewport:{width,height:width<1025?664:900}});
  let count=0,review=false;
  const state=()=>({admin:false,remaining:5-count+(review?5:0),limit:review?10:5,bonus_available:!review,feedback_submitted:review});
  await context.route('https://review.test/**',r=>{
   const p=new URL(r.request().url()).pathname;
   const json=(data,status=200)=>r.fulfill({status,contentType:'application/json',body:JSON.stringify(data)});
   if(p==='/me')return json({user:{id:1,name:'Посетитель',email:'test@example.invalid'}});
   if(p==='/account/usage')return json(state());
   if(p==='/conversations')return json({conversations:[]});
   if(p==='/ask/jobs'){count++;return json({job_id:'job'+count,conversation_id:'c'.repeat(43),status:'queued',ahead:0},202);}
   if(p.startsWith('/ask/jobs/'))return json({status:'done',answer:'Почва состоит из нескольких горизонтов.',conversation_id:'c'.repeat(43)});
   if(p==='/feedback'){
    const data=r.request().postDataJSON();assert.equal(data.stars,1);assert.equal(data.conversation_id,'c'.repeat(43));assert(data.comment.length>=10);review=true;return json(state(),201);
   }
   const file=path.join(__dirname,'../../static',p==='/'?'index.html':p);
   return fs.existsSync(file)&&fs.statSync(file).isFile()?r.fulfill({path:file}):r.fulfill({status:404,body:''});
  });
  const page=await context.newPage(),errors=[];page.on('pageerror',e=>errors.push(e.message));
  await page.goto('https://review.test/');await page.waitForFunction(()=>document.documentElement.dataset.chatReady==='true');
  const theme=page.locator('.theme-picker select:visible').first();await theme.selectOption('field');await page.reload();
  await page.waitForFunction(()=>document.documentElement.dataset.chatReady==='true');
  assert.equal(await page.locator('html').getAttribute('data-scene'),'field');
  if(process.env.REVIEW_SCREENSHOTS){fs.mkdirSync(process.env.REVIEW_SCREENSHOTS,{recursive:true});await page.screenshot({path:path.join(process.env.REVIEW_SCREENSHOTS,`field-${width}.png`)});}
  for(let i=0;i<3;i++){
   await page.locator('#chat-input').fill('Расскажи про почву');await page.locator('#send-btn').click();
   await page.waitForFunction(()=>!document.querySelector('#send-btn').disabled);
  }
  await page.locator('.feedback-card').waitFor();
  assert.match(await page.locator('#quota-status').innerText(),/2 из 5/);
  if(process.env.REVIEW_SCREENSHOTS)await page.screenshot({path:path.join(process.env.REVIEW_SCREENSHOTS,`feedback-${width}.png`)});
  await page.locator('.feedback-card summary').click();
  await page.locator('.feedback-card input[value="1"]').check();
  await page.locator('.feedback-card textarea').fill('Хотелось бы больше ссылок на источники.');
  await page.locator('.feedback-card button[type=submit]').click();
  await page.waitForFunction(()=>document.querySelector('#quota-status').textContent.includes('7 из 10'));
  assert.match(await page.locator('.feedback-card').innerText(),/Спасибо/);
  assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth),false);
  assert.deepEqual(errors,[]);console.log('PASS feedback bonus / one-star review / theme persistence / width',width);
  await context.close();
 }
 await browser.close();
})().catch(e=>{console.error(e);process.exit(1)});
