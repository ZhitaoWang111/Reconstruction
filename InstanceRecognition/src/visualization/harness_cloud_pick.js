/* Exact depth-tested object picking, using the same positions and camera as rendering. */
(function(){
  const canvas=document.getElementById('cloudCanvas');
  const info=document.getElementById('cloudPickInfo');
  let gesture=null, picker=null;
  function pick(clientX,clientY){
    if(!glState||!glState.pickIds)return null;
    const s=glState,gl=s.gl;
    resizeCanvasToDisplaySize(canvas);
    if(!picker){
      const program=createProgram(gl,`attribute vec3 a_position;attribute vec3 a_color;uniform mat4 u_matrix;uniform float u_pointSize;varying vec3 v_color;void main(){gl_Position=u_matrix*vec4(a_position,1.0);gl_PointSize=u_pointSize;v_color=a_color/255.0;}`,`precision mediump float;varying vec3 v_color;void main(){vec2 c=gl_PointCoord-vec2(.5);if(dot(c,c)>.25)discard;gl_FragColor=vec4(v_color,1.0);}`);
      picker={program,fb:gl.createFramebuffer(),texture:gl.createTexture(),depth:gl.createRenderbuffer(),colors:gl.createBuffer(),position:gl.getAttribLocation(program,'a_position'),color:gl.getAttribLocation(program,'a_color'),matrix:gl.getUniformLocation(program,'u_matrix'),size:gl.getUniformLocation(program,'u_pointSize')};
    }
    const p=picker;
    gl.bindTexture(gl.TEXTURE_2D,p.texture);gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MIN_FILTER,gl.NEAREST);gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MAG_FILTER,gl.NEAREST);
    gl.texImage2D(gl.TEXTURE_2D,0,gl.RGBA,canvas.width,canvas.height,0,gl.RGBA,gl.UNSIGNED_BYTE,null);
    gl.bindRenderbuffer(gl.RENDERBUFFER,p.depth);gl.renderbufferStorage(gl.RENDERBUFFER,gl.DEPTH_COMPONENT16,canvas.width,canvas.height);
    gl.bindFramebuffer(gl.FRAMEBUFFER,p.fb);gl.framebufferTexture2D(gl.FRAMEBUFFER,gl.COLOR_ATTACHMENT0,gl.TEXTURE_2D,p.texture,0);gl.framebufferRenderbuffer(gl.FRAMEBUFFER,gl.DEPTH_ATTACHMENT,gl.RENDERBUFFER,p.depth);
    const dither=gl.isEnabled(gl.DITHER),blend=gl.isEnabled(gl.BLEND);
    try{
      gl.disable(gl.DITHER);gl.disable(gl.BLEND);gl.enable(gl.DEPTH_TEST);gl.viewport(0,0,canvas.width,canvas.height);gl.clearColor(0,0,0,1);gl.clear(gl.COLOR_BUFFER_BIT|gl.DEPTH_BUFFER_BIT);gl.useProgram(p.program);gl.uniformMatrix4fv(p.matrix,false,computeMatrix(canvas.width/canvas.height));
      function draw(buffer,count,size,colors){
        gl.uniform1f(p.size,size*devicePixelRatio);gl.bindBuffer(gl.ARRAY_BUFFER,buffer);gl.enableVertexAttribArray(p.position);gl.vertexAttribPointer(p.position,3,gl.FLOAT,false,0,0);
        if(colors){gl.bindBuffer(gl.ARRAY_BUFFER,p.colors);gl.bufferData(gl.ARRAY_BUFFER,colors,gl.STREAM_DRAW);gl.enableVertexAttribArray(p.color);gl.vertexAttribPointer(p.color,3,gl.UNSIGNED_BYTE,false,0,0);}
        else{gl.disableVertexAttribArray(p.color);gl.vertexAttrib3f(p.color,0,0,0);}
        gl.drawArrays(gl.POINTS,0,count);
      }
      if(document.getElementById('showBackground').checked&&s.bgCount)draw(s.bgPositionBuffer,s.bgCount,1,null);
      const colors=new Uint8Array(s.pickIds.length*3);
      s.pickIds.forEach((id,i)=>{const value=id+1;colors[i*3]=value&255;colors[i*3+1]=(value>>8)&255;colors[i*3+2]=(value>>16)&255;});
      draw(s.positionBuffer,s.drawCount,3,colors);
      const r=canvas.getBoundingClientRect(),x=Math.floor((clientX-r.left)*canvas.width/r.width),y=canvas.height-1-Math.floor((clientY-r.top)*canvas.height/r.height);
      if(x<0||y<0||x>=canvas.width||y>=canvas.height)return null;
      // A small aperture makes sparse points clickable; prefer the closest visible pixel.
      const radius=Math.ceil(5*devicePixelRatio),x0=Math.max(0,x-radius),y0=Math.max(0,y-radius),w=Math.min(canvas.width-1,x+radius)-x0+1,h=Math.min(canvas.height-1,y+radius)-y0+1;
      const pixels=new Uint8Array(w*h*4);gl.readPixels(x0,y0,w,h,gl.RGBA,gl.UNSIGNED_BYTE,pixels);
      let best=null,dist=Infinity;
      for(let j=0;j<h;j++)for(let i=0;i<w;i++){const k=(j*w+i)*4,value=pixels[k]+pixels[k+1]*256+pixels[k+2]*65536,d=(x0+i-x)**2+(y0+j-y)**2;if(value&&d<dist){best=value-1;dist=d;}}
      return best;
    }finally{gl.bindFramebuffer(gl.FRAMEBUFFER,null);if(dither)gl.enable(gl.DITHER);if(blend)gl.enable(gl.BLEND);renderCloud();}
  }
  canvas.addEventListener('pointerdown',e=>{if(e.button===0)gesture={id:e.pointerId,x:e.clientX,y:e.clientY,moved:false};});
  canvas.addEventListener('pointermove',e=>{if(gesture&&e.pointerId===gesture.id&&Math.hypot(e.clientX-gesture.x,e.clientY-gesture.y)>5)gesture.moved=true;});
  canvas.addEventListener('pointerup',e=>{
    const g=gesture;gesture=null;if(!g||g.id!==e.pointerId||g.moved)return;
    const id=pick(e.clientX,e.clientY),object=stateCloud?.snapshot.objects.find(o=>o.id===id);
    info.textContent=object?`点击物体：${object.object_label} · Object ${String(object.id).padStart(3,'0')} · ${object.frames.length} 帧支持`:id===0?'当前点云为待定观测，尚未确定物体归属':'未命中已标注物体，请点击彩色点云。';
  });
  canvas.addEventListener('pointercancel',()=>{gesture=null;});
  const update=updateCloudBuffers;
  updateCloudBuffers=function(){update();info.textContent='点击点云查看类别名和物体编号';};
})();
