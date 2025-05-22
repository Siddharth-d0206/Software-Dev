// /static/js/firebase.js
import { initializeApp } from "https://www.gstatic.com/firebasejs/9.6.1/firebase-app.js";
import {
  getAuth,
  createUserWithEmailAndPassword,  
  signInWithEmailAndPassword,
  signOut
} from "https://www.gstatic.com/firebasejs/9.6.1/firebase-auth.js";
import {
  getFirestore,
  collection,
  addDoc
} from "https://www.gstatic.com/firebasejs/9.6.1/firebase-firestore.js";

const firebaseConfig = {
  apiKey: "AIzaSyCSs8RlFLTXofnvUcYYgoTFulJtGppWiBI",
  authDomain: "fitnessai-5ffc4.firebaseapp.com",
  projectId: "fitnessai-5ffc4",
  storageBucket: "fitnessai-5ffc4.appspot.com",
  messagingSenderId: "175181615870",
  appId: "1:175181615870:web:09ab675705e2b175c8cbaf",
  measurementId: "G-XTNEB3TBC3"
};

const app = initializeApp(firebaseConfig);
const auth = getAuth(app);
const db = getFirestore(app);

export {
  auth,
  db,
  getAuth, // <- this is required if you're importing `getAuth` directly elsewhere
  signOut,
  createUserWithEmailAndPassword,
  signInWithEmailAndPassword,
  collection,
  addDoc
};
