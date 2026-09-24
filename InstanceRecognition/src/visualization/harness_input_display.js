/* Show the saved API evidence, never a reconstruction from projected points. */
draw2d=function(){
  const img=document.getElementById('agentOverlay');
  const row=RESULT_FRAMES[currentFrameIndex];
  const path=row.agent_overlay;
  let note=document.getElementById('overlayProvenance');
  if(!note){note=document.createElement('p');note.id='overlayProvenance';img.parentNode.insertBefore(note,img);}
  note.textContent=row.overlay_note||'';
  note.hidden=!row.overlay_note;
  if(img.getAttribute('src')!==path)img.src=path;
  currentMaskBounds=[];
  document.getElementById('maskLabels').replaceChildren();
};
