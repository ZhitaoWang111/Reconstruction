/* Display-only overlay: each window uses boxes fitted from its committed cloud. */
(() => {
  const canvas = $('cloudCanvas'), layer = $('cloudObjectLabels');
  const labelsToggle = $('showObjectLabels'), boxesToggle = $('showObjectBboxes');
  let resources = null, cachedKey = '', vertexCount = 0;
  const edges = [];
  for (let i=0;i<8;i++) for (const bit of [1,2,4]) if (!(i&bit)) edges.push([i,i|bit]);
  function init(gl) {
    const program = createProgram(gl, `attribute vec3 a_position,a_other,a_color;
      attribute float a_side; uniform mat4 u_matrix; uniform vec2 u_viewport;
      varying vec3 v_color;
      void main(){
        vec4 p=u_matrix*vec4(a_position,1.0),q=u_matrix*vec4(a_other,1.0);
        float dp=p.z+p.w,dq=q.z+q.w;
        if(dp<0.0&&dq>=0.0)p=mix(p,q,dp/(dp-dq));
        else if(dq<0.0&&dp>=0.0)q=mix(q,p,dq/(dq-dp));
        vec2 d=(q.xy/max(q.w,0.000001)-p.xy/max(p.w,0.000001))*u_viewport;
        vec2 n=vec2(-d.y,d.x)/max(length(d),0.000001);
        p.xy+=n*a_side/u_viewport*p.w;gl_Position=p;v_color=a_color/255.0;
      }`, `precision mediump float;varying vec3 v_color;void main(){gl_FragColor=vec4(v_color,1.0);}`);
    return {program, attributes:['position','other','side','color'].map(name=>({
      buffer:gl.createBuffer(),location:gl.getAttribLocation(program,'a_'+name),size:name==='side'?1:3
    })), matrix:gl.getUniformLocation(program,'u_matrix'), viewport:gl.getUniformLocation(program,'u_viewport')};
  }
  function upload(gl, objects) {
    const arrays=[[],[],[],[]];
    for(const {object,box} of objects) for(const [a,b] of edges)
      for(const [from,to,side] of [[a,b,-1],[a,b,1],[b,a,1],[b,a,1],[a,b,1],[b,a,-1]]) {
        arrays[0].push(...box.corners_xyz[from]);arrays[1].push(...box.corners_xyz[to]);
        arrays[2].push(side*3);arrays[3].push(...object.color);
      }
    vertexCount=arrays[2].length;
    resources.attributes.forEach((a,i)=>{gl.bindBuffer(gl.ARRAY_BUFFER,a.buffer);gl.bufferData(gl.ARRAY_BUFFER,new Float32Array(arrays[i]),gl.STATIC_DRAW);});
  }
  const baseRender=renderCloud;
  renderCloud=function(){
    baseRender();layer.replaceChildren();
    if(!glState || !stateCloud || (!labelsToggle.checked&&!boxesToggle.checked))return;
    const windowIndex=RESULT_FRAMES[currentFrameIndex].window_index;
    const boxes=window.HARNESS_BBOX_WINDOWS?.[windowIndex] || {};
    const objects=stateCloud.snapshot.objects.filter(o=>objectIsChecked(o.id))
      .map(object=>({object,box:boxes[String(object.id)]})).filter(x=>x.box?.corners_xyz?.length===8);
    const gl=glState.gl,matrix=computeMatrix(canvas.width/canvas.height);
    if(boxesToggle.checked && objects.length){
      if(!resources)resources=init(gl);
      const key=windowIndex+':'+objects.map(x=>x.object.id).join(',');
      if(key!==cachedKey){upload(gl,objects);cachedKey=key;}
      gl.useProgram(resources.program);gl.uniformMatrix4fv(resources.matrix,false,matrix);
      gl.uniform2f(resources.viewport,canvas.clientWidth,canvas.clientHeight);
      resources.attributes.forEach(a=>{gl.bindBuffer(gl.ARRAY_BUFFER,a.buffer);gl.enableVertexAttribArray(a.location);gl.vertexAttribPointer(a.location,a.size,gl.FLOAT,false,0,0);});
      gl.drawArrays(gl.TRIANGLES,0,vertexCount);
      // Restore attribute state before the existing point renderer or picker runs.
      resources.attributes.forEach(a=>gl.disableVertexAttribArray(a.location));
      gl.useProgram(glState.program);
    }
    if(labelsToggle.checked)for(const {object,box} of objects){
      const p=box.center_xyz;
      const q=[0,1,2,3].map(r=>matrix[r]*p[0]+matrix[4+r]*p[1]+matrix[8+r]*p[2]+matrix[12+r]);
      if(q[3]<=0 || q.slice(0,3).some(v=>Math.abs(v)>q[3]))continue;
      const label=document.createElement('span');label.className='cloudObjectLabel';label.dataset.objectId=object.id;
      label.textContent=`Object ${String(object.id).padStart(3,'0')} · ${object.object_label}`;
      label.style.left=`${(q[0]/q[3]+1)*50}%`;label.style.top=`${(1-q[1]/q[3])*50}%`;
      label.style.borderColor=`rgb(${object.color.join(',')})`;layer.appendChild(label);
    }
  };
  labelsToggle.onchange=boxesToggle.onchange=()=>renderCloud();
})();
