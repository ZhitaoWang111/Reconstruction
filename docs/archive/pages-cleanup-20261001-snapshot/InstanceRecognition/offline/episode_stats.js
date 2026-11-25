(function exposeEpisodeStats(root) {
  function computeEpisodeStats(scene, confidenceThreshold, minimumFrames = 1) {
    const byClass = new Map();
    scene.frames.forEach((frame, framePosition) => {
      const seenInFrame = new Set();
      for (const instance of frame.instances) {
        if (instance.confidence < confidenceThreshold) continue;
        const name = instance.class_name;
        if (!byClass.has(name)) {
          byClass.set(name, {
            class_name: name,
            detected_frames: 0,
            instance_count: 0,
            confidence_sum: 0,
            maximum_confidence: -Infinity,
            representative_frame: frame.frame_index,
            representative_position: framePosition,
          });
        }
        const item = byClass.get(name);
        item.instance_count += 1;
        item.confidence_sum += instance.confidence;
        if (instance.confidence > item.maximum_confidence) {
          item.maximum_confidence = instance.confidence;
          item.representative_frame = frame.frame_index;
          item.representative_position = framePosition;
        }
        seenInFrame.add(name);
      }
      for (const name of seenInFrame) byClass.get(name).detected_frames += 1;
    });

    const classes = [...byClass.values()]
      .filter(item => item.detected_frames >= minimumFrames)
      .map(item => ({
        ...item,
        mean_confidence: item.confidence_sum / item.instance_count,
        frame_ratio: item.detected_frames / scene.frame_count,
      }))
      .sort((left, right) => (
        right.detected_frames - left.detected_frames
        || right.instance_count - left.instance_count
        || right.maximum_confidence - left.maximum_confidence
        || left.class_name.localeCompare(right.class_name)
      ));
    const retainedNames = new Set(classes.map(item => item.class_name));
    const detectedFrames = scene.frames.filter(frame => frame.instances.some(instance => (
      instance.confidence >= confidenceThreshold
      && retainedNames.has(instance.class_name)
    ))).length;

    return {
      classes,
      class_count: classes.length,
      instance_count: classes.reduce((sum, item) => sum + item.instance_count, 0),
      detected_frames: detectedFrames,
    };
  }

  root.computeEpisodeStats = computeEpisodeStats;
  if (typeof module !== 'undefined' && module.exports) {
    module.exports = {computeEpisodeStats};
  }
}(globalThis));
