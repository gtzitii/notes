<script setup>
import HelloWorld from './components/HelloWorld.vue'

import axios from 'axios'
import { ref } from 'vue'

const text = ref('')  // 用来保存后端返回的数据
const visitToBackend = async () => {
  try {
    const res = await axios.get('/api/hello')  // 访问后端接口
    text.value = res.data   // 更新显示
  } catch (error) {
    console.error('Error fetching data from backend:', error)
    text.value = '请求失败'
  }
}

</script>

<template>
  <div>
    <a href="https://vite.dev" target="_blank">
      <img src="/vite.svg" class="logo" alt="Vite logo" />
    </a>
    <a href="https://vuejs.org/" target="_blank">
      <img src="./assets/vue.svg" class="logo vue" alt="Vue logo" />
    </a>
  </div>
  <HelloWorld msg="Vite + Vue" />

  <!-- 点击按钮访问后端 -->
  <button @click="visitToBackend">访问后端</button>

  <!-- 展示后端返回的数据 -->
  <input v-model="text" />
  
</template>

<style scoped>
.logo {
  height: 6em;
  padding: 1.5em;
  will-change: filter;
  transition: filter 300ms;
}
.logo:hover {
  filter: drop-shadow(0 0 2em #646cffaa);
}
.logo.vue:hover {
  filter: drop-shadow(0 0 2em #42b883aa);
}
</style>
