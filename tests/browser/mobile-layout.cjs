/* Run with Playwright installed: node tests/browser/mobile-layout.cjs.
 * Uses local static assets and a mock API; it never contacts the public site.
 * Device profiles include browser chrome (390x664, rather than full 390x844).
 */
const { chromium, devices } = require('playwright');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const STATIC = path.resolve(__dirname, '../../static');
const OUT = process.env.MOBILE_SCREENSHOT_DIR;
if (OUT) fs.mkdirSync(OUT, {recursive:true});

(async () => {
  const browser = await chromium.launch({executablePath:process.env.CHROMIUM_PATH || '/usr/bin/chromium',args:['--no-sandbox']});
  const profiles = [
    ['iphone', {...devices['iPhone 13']}],
    ['pixel', {...devices['Pixel 7']}],
    ['small', {...devices['iPhone SE']}],
    ['landscape', {...devices['iPhone 13 landscape']}],
    ['wide-landscape', {...devices['iPhone 13'],viewport:{width:844,height:390}}],
    ['tablet', {viewport:{width:1024,height:768},isMobile:true,hasTouch:true,deviceScaleFactor:2}],
    ['desktop', {viewport:{width:1440,height:900}}],
    ['compact-desktop', {viewport:{width:1100,height:900}}],
    ['wide-desktop', {viewport:{width:1920,height:1080}}],
    ['ultrawide-desktop', {viewport:{width:2560,height:1440}}],
  ];
  for (const [name, profile] of profiles.filter(([name])=>!process.env.MOBILE_PROFILE || name===process.env.MOBILE_PROFILE)) {
    const context = await browser.newContext(profile);
    let user = null, phase='processing', requests=0;
    const errors=[];
    await context.route('https://mobile.test/**', route => {
      const req=route.request(), uri=new URL(req.url()), p=uri.pathname;
      const reply=(data,status=200)=>route.fulfill({status,contentType:'application/json',body:JSON.stringify(data)});
      if(p==='/me')return reply(user?{user}:{detail:'Войдите'},user?200:401);
      if(p==='/register'||p==='/login') { user={id:1,name:'Очень длинное имя посетителя музея',email:'test@example.invalid'};return reply({user}); }
      if(p==='/account/password') {const body=req.postDataJSON();return body.current_password==='test-password-123'?reply({user}):reply({detail:'Проверьте текущий пароль'},403);}
      if(p==='/logout') { user=null;return reply({},204); }
      if(p==='/conversations')return reply({conversations:[{conversation_id:'c'.repeat(43),title:'Сохранённая беседа о почве',updated_at:'2026-11-05T08:00:00Z'}]});
      if(p.startsWith('/conversations/'))return req.method()==='DELETE'?reply({},204):reply({conversation_id:'c'.repeat(43),turns:[{question:'Какая почва?',answer:'Чернозём.',has_image:false}],active_job_id:null});
      if(p==='/ask/jobs') { requests++;return reply({job_id:'j'+requests,conversation_id:'c'.repeat(43),status:'queued',ahead:2},202); }
      if(p.startsWith('/ask/jobs/'))return reply({job_id:'j'+requests,conversation_id:'c'.repeat(43),status:phase,ahead:2,answer:'Готовое объяснение по вопросу о почве.'});
      const file=path.join(STATIC,p==='/'?'index.html':p);
      return fs.existsSync(file)&&fs.statSync(file).isFile()?route.fulfill({path:file}):route.fulfill({status:404,body:''});
    });
    const page=await context.newPage();page.on('pageerror',e=>errors.push(e.message));
    await page.goto('https://mobile.test/');await page.waitForFunction(()=>document.documentElement.dataset.chatReady==='true');
    const mobile=profile.viewport.width<=1024;
    async function fits() {
      assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth+1),false,`${name}: page spills horizontally`);
      assert.equal(await page.evaluate(()=>visualViewport.scale),1,`${name}: browser auto-zooms`);
      for(const sel of ['#chat-input','#attach-photo','#send-btn']) {
        const b=await page.locator(sel).boundingBox();assert(b&&b.width>=20&&b.x>=0&&b.x+b.width<=profile.viewport.width+1,`${name}: ${sel} clipped`);
      }
    }
    await fits();
    if(!mobile) {
      const main=await page.locator('.main-content').boundingBox();
      const nav=await page.locator('.sidebar').boundingBox();
      assert(Math.abs(main.x+main.width/2-profile.viewport.width/2)<2,`${name}: content off centre`);
      assert(main.width<=1320 && main.x>=31,`${name}: forest margins missing`);
      assert(Math.abs(nav.x-main.x)<2 && Math.abs(nav.width-main.width)<2,`${name}: navigation misaligned`);
      const links=await page.locator('.sidebar .nav-item').all();
      for(const link of links) {
        const box=await link.boundingBox();
        assert(box.x>=nav.x && box.x+box.width<=nav.x+nav.width,`${name}: navigation clipped`);
      }
      if(OUT)await page.screenshot({path:path.join(OUT,name+'-catalog.png'),scale:'css'});
    }
    if(mobile) {
      assert.equal(await page.locator('#catalog').isVisible(),false);
      assert.equal(await page.locator('#ai-chat').isVisible(),true);
      const composer=await page.locator('.input-area').boundingBox();
      if(profile.viewport.height>=550)assert(composer.y+composer.height<=profile.viewport.height,`${name}: composer below browser viewport`);
      assert.equal(await page.locator('#burger-menu').isVisible(),true);
      await page.locator('#burger-menu')[mobile ? 'tap' : 'click']();await page.waitForSelector('.sidebar.visible');
      await page.locator('#overlay')[mobile ? 'tap' : 'click']({position:{x:profile.viewport.width-5,y:profile.viewport.height-5}});
      assert.equal(await page.locator('#burger-menu').getAttribute('aria-expanded'),'false');
      await page.locator('.mobile-section-nav a[href="#catalog"]')[mobile ? 'tap' : 'click']();
      assert.equal(await page.locator('#ai-chat').isVisible(),false);
      assert.equal(await page.locator('.soil-card:visible').count(),6);
      await page.waitForFunction(()=>{const img=document.querySelector('.soil-card:not([hidden]) img');return img.complete&&img.naturalWidth>0;});
      if(OUT) await page.screenshot({path:path.join(OUT,name+'-catalog.png'),scale:'css'});
      const first=await page.locator('.soil-card:visible').first().boundingBox();assert(first.width>profile.viewport.width*.65,`${name}: squeezed card`);
      const a=await page.locator('.soil-card:visible').nth(1).boundingBox();assert(a.y>first.y,`${name}: unexpected horizontal carousel`);
      await page.locator('#catalog-search').fill('Грузия');assert.equal(await page.locator('.soil-card:visible').count(),1);
      await page.locator('#catalog-reset')[mobile ? 'tap' : 'click']();await page.locator('#catalog-more')[mobile ? 'tap' : 'click']();assert.equal(await page.locator('.soil-card:visible').count(),12);
      await page.locator('.mobile-chat-link')[mobile ? 'tap' : 'click']();await page.goBack();assert.equal(await page.locator('#catalog').isVisible(),true);
      await page.locator('.mobile-chat-link')[mobile ? 'tap' : 'click']();await page.evaluate(()=>scrollTo(0,0));
      if(OUT)await page.screenshot({path:path.join(OUT,name+'-chat.png'),scale:'css'});
    }
    await page.locator('#open-auth')[mobile ? 'tap' : 'click']();await page.locator('#auth-register-tab')[mobile ? 'tap' : 'click']();
    const dialog=await page.locator('#auth-dialog').boundingBox();assert(dialog.x>=0&&dialog.x+dialog.width<=profile.viewport.width+1,`${name}: dialog width`);
    assert(Math.abs(dialog.x+dialog.width/2-profile.viewport.width/2)<2,`${name}: dialog off centre`);
    if (mobile && profile.viewport.height >= 550) {
      // Simulate the visible area shrinking without changing the layout viewport.
      const keyboardBox = await page.evaluate(() => {
        const vv = visualViewport;
        Object.defineProperty(vv, 'height', {configurable:true, get:()=>300});
        vv.dispatchEvent(new Event('resize'));
        const box = document.querySelector('#auth-dialog').getBoundingClientRect();
        return {top:box.top, bottom:box.bottom, offset:vv.offsetTop};
      });
      assert(keyboardBox.top >= keyboardBox.offset && keyboardBox.bottom <= keyboardBox.offset+300, name+': dialog under keyboard');
      await page.evaluate(() => { delete visualViewport.height;visualViewport.dispatchEvent(new Event('resize')); });
    }
    await page.locator('#auth-name').fill('Имя');await page.locator('#auth-email').fill('test@example.invalid');await page.locator('#auth-password').fill('test-password-123');await page.locator('#auth-terms').check();await page.locator('#auth-submit')[mobile ? 'tap' : 'click']();
    await page.waitForFunction(()=>!document.querySelector('#auth-dialog').open);await fits();
    if(mobile) {
      await page.locator('#mobile-chats-toggle')[mobile ? 'tap' : 'click']();await page.locator('.saved-chat-open')[mobile ? 'tap' : 'click']();
      assert.equal(await page.locator('#saved-chat-list').isVisible(),false);
      await page.locator('#memory-toggle')[mobile ? 'tap' : 'click']();assert.equal(await page.locator('#memory-note').isVisible(),true);await page.locator('#memory-toggle')[mobile ? 'tap' : 'click']();
    }
    await page.locator('#chat-input').fill('Объясни свойства почвы');await page.locator('#send-btn')[mobile ? 'tap' : 'click']();
    await page.locator('[data-wait-state="processing"]').waitFor();
    assert.equal(await page.locator('#new-chat').isVisible(),true, name+': new-chat inaccessible');
    if(OUT) {await page.locator('.soil-wait').scrollIntoViewIfNeeded();await page.screenshot({path:path.join(OUT,name+'-waiting.png'),scale:'css'});}
    const w=await page.locator('#chat-window').boundingBox(),loader=await page.locator('.soil-wait').boundingBox();
    assert(loader.x>=w.x&&loader.x+loader.width<=w.x+w.width+1,`${name}: waiting too wide`);
    if(profile.viewport.height>=550) assert(loader.height<=w.height,`${name}: waiting too tall`);
    phase='done';await page.locator('.message.ai').filter({hasText:'Готовое объяснение'}).waitFor();
    if (mobile && profile.viewport.height>=550) {
      const b=await page.locator('.input-area').boundingBox();
      assert(b.y+b.height<=profile.viewport.height+1, name+': member composer off screen');
    }
    assert.equal(await page.locator('#send-btn').getAttribute('aria-label'),'Отправить сообщение');
    if(mobile) {
      await page.locator('#chat-input').focus();
      await page.setViewportSize({width:profile.viewport.width,height:350});await page.waitForTimeout(100);
      await fits();
      await page.setViewportSize(profile.viewport);
    }
    await page.locator('#open-security')[mobile?'tap':'click']();
    await page.locator('#current-password').fill('wrong-password');
    await page.locator('#new-password').fill('New-unique-phrase-for-site-52');
    await page.locator('#confirm-password').fill('Another-unique-phrase-63');
    await page.locator('#change-password-submit')[mobile?'tap':'click']();
    await page.locator('#password-error').filter({hasText:'не совпадают'}).waitFor();
    await page.locator('#confirm-password').fill('New-unique-phrase-for-site-52');
    await page.locator('#change-password-submit')[mobile?'tap':'click']();
    await page.locator('#password-error').filter({hasText:'текущий пароль'}).waitFor();
    await page.locator('#current-password').fill('test-password-123');
    await page.locator('#change-password-submit')[mobile?'tap':'click']();
    await page.waitForFunction(()=>!document.querySelector('#security-dialog').open);
    assert((await page.locator('#chat-notice').textContent()).includes('Пароль изменён'));
    await page.locator('#file-upload').setInputFiles(path.join(STATIC,'images/soils/013/monolith.jpg'));
    await page.locator('#photo-preview:not([hidden])').waitFor();
    await fits();
    await page.locator('#send-btn')[mobile?'tap':'click']();
    await page.waitForFunction(()=>!document.querySelector('#send-btn').disabled);
    assert.equal(await page.locator('#photo-preview').isVisible(),false,name+': photo not submitted');
    assert.deepEqual(errors,[],name+' script errors');console.log('PASS',name,profile.viewport,'navigation / catalog / account / composer / waiting');
    await context.close();
  }
  await browser.close();
})().catch(e=>{console.error(e);process.exit(1);});
